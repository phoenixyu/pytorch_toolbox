# Pytorch Toolbox For Segmentation (V0.1)

## Introduction
This repo is used for training&testing deep learning problems(segmentation). It is implemented by Python & Pytorch. (Some code from others to share, thank you for their efforts!)

## Dependency
* [Python3.6](https://www.python.org/downloads/)
* [Numpy](http://www.numpy.org/)
* [OpenCV3](https://opencv.org/)
* [Pytorch](http://pytorch.org/docs/master/)
* [Torchvision](https://pypi.python.org/pypi/torchvision/)
* [Sklearn](http://scikit-learn.org/stable/)
* [tqdm](https://github.com/tqdm/tqdm)

## Usage
### 1. Preprocess
Open utils/generate_data.py and edit relevent info.
```python
ProjectDir = "/home/dl/phoenix_lzx/torch/data"
crop_size = 320
training_data_stage1_dir = os.path.join(ProjectDir,"seaship")

img_list_1=[os.path.join(training_data_stage1_dir,'{}.jpg'.format(item)) for item in img_file_list]
label_list_1=[os.path.join(training_data_stage1_dir,'{}.png'.format(item)) for item in img_file_list]

dataset_dir=os.path.join(ProjectDir,"dataset/seaship-train")
```

### 2. Dataloader
Edit utils/seaship_loader.py and set it to fit your dataset.

### 3.Set model
Edit models/init.py, now this repo provide some models below.(Some code from others to share, thank you for their efforts!)
- Alexnet
- UNet
- DeepUNet
- FCN
- [RefineNet](https://github.com/thomasjpfan/pytorch_refinenet)
- SegNet

### 4.Change Train param.
Edit main.py to complete the setting. Then just RUN it.

### 5.Test
Edit test.py to fit your testset.

## Help
This repo has just started, some problems needed to be solved. If you find it, please contact me immediately.