import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import FCNet
import torch
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error,mean_absolute_error
from torchvision import transforms
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import torch.nn.functional as FF
from thop import profile
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]



class resizeImage(object):
    def __init__(self, MAX_HW=600):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image = sample['image']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image
    
        resized_image = Normalize(resized_image)
        sample = {'image':resized_image}
        return sample

Normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])
Transform = transforms.Compose([resizeImage(600)])


def format_for_plotting(tensor):
    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()


def denormalize(tensor, means = IM_NORM_MEAN, stds = IM_NORM_STD):
    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)

    return denormalized


def scale_and_clip(val, scale_factor, min_val, max_val):
    new_val = int(round(val*scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val


def visualize_output_and_save(gt, pt, input_, output, save_path, figsize=(20, 12)):
    # get the total count
    pred_cnt = pt
    img1 = format_for_plotting(denormalize(input_))
    output = format_for_plotting(output)
    fig = plt.figure(figsize=figsize)  

    # display the input image

    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.set_title("gt count: {}".format(int(gt)))
    img2 = 0.2989 * img1[:, :, 0] + 0.5870 * img1[:, :, 1] + 0.1140 * img1[:, :, 2]
    ax.imshow(img2, cmap='gray')
    # ax.imshow(gt, cmap=plt.cm.viridis, alpha=0.5)


    #Overlaid result
    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title("predicted count: {:.2f}".format(pred_cnt))
    img2 = 0.2989*img1[:, :, 0] + 0.5870*img1[:, :, 1] + 0.1140*img1[:, :, 2]  
    print(img2.shape, output.shape)
    ax.imshow(img2, cmap='gray')
    ax.imshow(output, cmap=plt.cm.jet, alpha=0.5)


    # display the density map
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output, cmap=plt.cm.jet)
    # plt.colorbar()

    ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ret_fig = ax.imshow(output, plt.cm.jet)

    fig.colorbar(ret_fig, ax=ax)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close()

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

parser = argparse.ArgumentParser(description='PyTorch CANNet')

parser.add_argument('--test_json', type=str, default='./datasets/density_val_list.json', metavar='test',
                    help='path to val json')

parser.add_argument('--output',type=str, default="./saved_models_2", metavar='TEST',
                    help='path output')

args = parser.parse_args()


with open(args.test_json, 'r') as outfile:
    img_paths = json.load(outfile)

model = FCNet()

model = model.cuda()

checkpoint = torch.load(os.path.join(args.output,'model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

pred= []
gt = []
t_all = []

mae=0
rmse=0
nae=0

for i in range(len(img_paths)):
    t1 = time.time()
    image = Image.open(img_paths[i])
    image.load()
    sample = {'image': image}
    sample = Transform(sample)
    image = sample['image'].cuda()
    image = image.unsqueeze(0)
    
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    img = img.unsqueeze(0)

   
    density = model(img, "test").data.cpu().numpy()
#     flops, params = profile(model, inputs=(img,"test"))
#     print("Params：", str(params/(1000 ** 2))+"M")
#     print("FLOPS：", str(flops/(1000 ** 3))+"G")
    pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5'))
    groundtruth = np.asarray(gt_file['density'])
    
    pred_sum = density.sum()
    t2 = time.time()
    t_all.append(t2 - t1)
    
    mae+=abs(pred_sum-np.sum(groundtruth)).item()
    nae+=(abs(pred_sum-np.sum(groundtruth))/np.sum(groundtruth)).item()
    rmse+=abs(pred_sum-np.sum(groundtruth)).item()**2

    
    gt = np.sum(np.asarray(gt_file['density']))
    pt = density.sum()
    density=torch.from_numpy(density)
#     output = FF.interpolate(density, size=(450,600), mode='bilinear')
#     rslt_file = "{}/{}_e4myout.png".format("val_r", img_paths[i][-7:-4])
#     visualize_output_and_save(gt, pt, image.detach().cpu(), output.detach().cpu(), rslt_file)


print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))


print('MAE: ',mae/len(img_paths))
print('RMSE: ',(rmse/len(img_paths))**0.5)
print('NAE: ',nae/len(img_paths))

results=np.array([mae,rmse])
np.savetxt(os.path.join(args.output,"restults.txt"),results,delimiter=',')
