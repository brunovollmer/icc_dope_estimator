# Movement Coach Pose Estimation

Pose Estimation model for the ICC 2020 submission of the Movement Coach team.

### Getting started

Our python3 code requires the following packages:
* pytorch
* torchvision
* opencv (for drawing the results)
* numpy/scipy

#### Download the models

First create a folder `models/` in which you should place the downloaded pretrained models.
The list of models include:
* [DOPE_v1_0_0](http://download.europe.naverlabs.com/ComputerVision/DOPE_models/DOPE_v1_0_0.pth.tgz) as used in our ECCV'20 paper
* [DOPErealtime_v1_0_0](http://download.europe.naverlabs.com/ComputerVision/DOPE_models/DOPErealtime_v1_0_0.pth.tgz) which is its real-time version


## Using the code

To use our code on an image, use the following command:

```
python dope.py --model <modelname> --image <imagename>
```

with
* `<modelname>`: name of model to use (eg DOPE_v1_0_0)
* `<imagename>`: name of the image to test

For instance, you can run
```
python dope.py  --model DOPErealtime_v1_0_0 --image 015994080.jpg
```

For videos run the following command:

```
python dope.py --model <modelname> --video <videoname>
```
with
* `<modelname>`: name of model to use (eg DOPE_v1_0_0)
* `<videoname>`: name of the video to test

Our real-time models use half computation. In case your device cannot handle it, please add --no_half_comp flag!
