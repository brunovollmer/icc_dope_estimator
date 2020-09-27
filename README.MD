# Movement Coach Pose Estimation

Pose Estimation model for the [ICC 2020](https://www.createchallenge.org/) submission of the Movement Coach team. The second part of the final submission can be found in [this repository](https://github.com/brunovollmer/icc_dope_server).

The pose estimation model is based on the Paper "DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild" and their [code](https://github.com/naver/dope).

```
@inproceedings{dope,
  title={{DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild}},
  author={{Weinzaepfel, Philippe and Br\'egier, Romain and Combaluzier, Hadrien and Leroy, Vincent and Rogez, Gr\'egory},
  booktitle={{ECCV}},
  year={2020}
}
```

Please consider that this prototype was created during a hackathon with the main focus on speed. Code and comment quality may vary :)

## License

The code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.


## Getting started

Create a virtual python environment and install all required packages.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download the models

Create a folder `models/` in which you should place the downloaded pretrained models. All pretrained models are supplied by the original authors of DOPE.
The list of models include:
* [DOPE_v1_0_0](http://download.europe.naverlabs.com/ComputerVision/DOPE_models/DOPE_v1_0_0.pth.tgz) as used in our ECCV'20 paper
* [DOPErealtime_v1_0_0](http://download.europe.naverlabs.com/ComputerVision/DOPE_models/DOPErealtime_v1_0_0.pth.tgz) which is its real-time version

## Using the code

Currently a few different options are supported to run the code. 

### Image
To use our code on an image, use the following command:

```
python main.py --model <modelname> --image <imagename>
```

with
* `<modelname>`: name of model to use (eg DOPE_v1_0_0)
* `<imagename>`: name of the image to test

For instance, you can run
```
python main.py  --model DOPErealtime_v1_0_0 --image 015994080.jpg
```

### Video
If you want to compare two videos run the following command:

```
python  main.py --model <modelname> --video <videoname> -m_video <mastervideoname>
```
with
* `<modelname>`: name of model to use (eg DOPE_v1_0_0)
* `<videoname>`: name of the user video
* `<mastervideoname>`: name of master video

### Config

Our real-time models use half computation. In case your device cannot handle it, please add `--no_half_comp` flag!

If you want to visualize or save the visualized results add the `--visualize` and\or `--save_images` flag!


