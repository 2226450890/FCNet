# FCNet
![image](https://github.com/2226450890/FCNet/blob/main/model1.jpg)
Accurate counting model of fish in complex underwater environments.
![image](https://github.com/2226450890/FCNet/blob/main/model2.jpg)
Feature extraction module, (a) represents the process of extracting interference features,
(b) represents the process of extracting counting features, (c) represents the process of intercepting
the background image, and (d) represents the process of intercepting the single fish image.
![image](https://github.com/2226450890/FCNet/blob/main/model3.jpg)
Counting feature extraction module and feature supplement module.

## Dataset
![image](https://github.com/2226450890/FCNet/blob/main/test1.jpg)
Download Underwater_Fish_2024 Dataset from
Mega: [link](https://mega.nz/file/vN92jDCL#aFKNgaLo1JK3Z8otlrCQ5zdaA-9pehfA7N57Rw8fIqw) 

## Counting results
![image](https://github.com/2226450890/FCNet/blob/main/test2.jpg)
Visualization of counting results. The original images, point annotation images and
predicted density maps are listed from top to bottom.

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.8

PyTorch: 1.11

CUDA: 11.3

## Evaluation
&emsp;1. We are providing our pretrained model, and the evaluation code can be used without the training. Download pretrained model from OneDrive: [link](https://stuscaueducn-my.sharepoint.com/:u:/g/personal/3170062_stu_scau_edu_cn/EV5-CSBgb2NPmBfWz_ks9woBKOb5vc42cW4BG6IfWQF4rQ?e=ydp6FM).

&emsp;2. Modify the model directory in the test. py file.

&emsp;3. Evaluate the model.
```
python test.py
```

## Training
&emsp;1. Configure the files required to run, and modify the root path in the "train.py" based on your dataset location.  

&emsp;2. Run train.py.
```
python train.py
```  

