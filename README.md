# Advanced RealTime Road Detection and Autonomous Vehicle Control using Convolutional Neural Networks and Human Learned Driving Models. 

## Intro
The goal of this project is to make an electric ATV drive autonomosly over normal roads using an xbox kinect Camera and state of the art video processing technolgies. The electric ATV is fitted with an windshield wiper motor and costum control/feedback circuitry to control the steering and acceleration. An Arduino with Serial comminucation is used to communicate with the main computer also fitted on the ATV. The main computer has an GTX1080 to run the detection model. The Detection model and Control module are written in python. 

While working on this project I discoverd two approaches for autonomous driving. The first is vision based which uses Ai to detect the road and use vision to calculate the centerline from the road detection. Then use standard control techniques like PID or MPC to make the center of the vehicle follow the centerline. This method is now discontinued in this project because there is a much more exciting way!

Instead of making complicated modules which can accuratly calculate the centerline and keep the center in all situations (practicly impossible because roads are always diffrent so using preprogrammed modules will only work 99% of the time), I use a second neural network to make driving decisions based on my own experience. This works in the following manner: 

- I drive the atv myself (doing throttle and steering manually) over the road. While doing this I save every frame the segmentated image from the nvidia model and the steering and throttle value at that moment. (range from -1 to 1 for steering and (0 to 1) for throttle.

- After gathering the data a AI model is trained to perform the same actions as the human did in the data.

- The resulting model is a robust model which exactly drives as a human. 

This Ai method works much better because driving is a really complicated task. It can not be performed by using some advanced control methods because driving is not just always a problem where the solution can be found with formulas. Sometimes human intuition is the only way to make the right decision. Because the Ai method learns from a real human driver which has this intuition the Ai also learns that human intuition once given enough data. This makes the Ai model really robust in every situation it is trained on. And thus when given enough data (Hunderds of millions of hours of driving data) the model will be able to handle every situation. 

Look at the GIF below for a model trained on 15 min of driving the ATV on a straight road. 

## Ai Based
![](https://github.com/sieuwe1/Advanced-Real-Time-Self-Driving/blob/master/Ai_based/DemoGif.gif)

- Red line Angle = Ai predicted steering
- Red line length = Ai predicted throttle 
- White area = Road
- Bleu area = Ground
- green area = trees and plants
- Black area = Uninteresting object classes

I have already perfomred some real runs with the ATV in autonomous mode. I will soon post some videos of the results! 

## Vision Based
![](https://github.com/sieuwe1/Advanced-Real-Time-Self-Driving/blob/master/Vision_based/demo.gif)

### How is Detection and Control working? 
- NVIDIA'S semantic-segmentation model returns every 10th frame an array with classes per pixel
- Road edges are extracted from the array with connected component filter
- Road center line is calculated from road edges with moving average filter
- Center of the frame is used as center of the vehicle to calcualte Delta between vehicle center and road center. 
- PID controller is used to generate steering commands from Delta
- Any obstacles which are on the vehucle path are detected with Kinect depth Camera.
- If obstacles are detected a speed reduce command is made
- Steering and speed commands are send over Serial to Arduino
- Arduino performs commands

## Based on the great work of
https://github.com/NVIDIA/semantic-segmentation/

# Installation
- Download pretrained network from link. 
https://drive.google.com/file/d/1aGdA1WAKKkU2y-87wSOE1prwrIzs_L-h/view

- Make an folder called "pretrained_models" and copy/paste network into that folder.
- install dependencies
* An NVIDIA GPU and CUDA 9.0 or higher. Some operations only have gpu implementation.
* PyTorch (>= 0.5.1)
* Python 3
* numpy
* OpenCV 
* sklearn
* h5py
* scikit-image
* pillow
* piexif
* cffi
* tqdm
* dominate
* tensorboardX
* opencv-python
* nose
* ninja
* pandas

# Run
For a quick demo to test if the envoirement is setup correctly use the below command. This will run both a pretrained drive model together with the segmentation model but this wont try to make a connection with the ATV. So this can just be ran without any hardware connected. 
```
CUDA_VISIBLE_DEVICES=0 python3 SegmentationOnly.py --video .
```

For a complete run with a ATV run this:
```
CUDA_VISIBLE_DEVICES=0 python3 driveAutonomous.py 
```

To train with your own data first drive a lot of traning runs with this command: 
```
 python3 Train.py --model=trained --type=linear
```

Make sure you first callibrated your steering and throttle sensors using the steps provided in the 
```
Advanced-Real-Time-Self-Driving/Ai_based/how to calibrate.txt
```
text file. 

Then copy paste the data folders to the folder  
```
Advanced-Real-Time-Self-Driving/Ai_based/donkeycar/data/
```

Finally run this command to start training a model with linear activation function. 
```
 python3 Train.py --model=trained --type=linear
```

# NVIDIA'S semantic-segmentation

@inproceedings{semantic_cvpr19,
  author       = {Yi Zhu*, Karan Sapra*, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao, Bryan Catanzaro},
  title        = {Improving Semantic Segmentation via Video Propagation and Label Relaxation},
  booktitle    = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month        = {June},
  year         = {2019},
  url          = {https://nv-adlr.github.io/publication/2018-Segmentation}
}
* indicates equal contribution

@inproceedings{reda2018sdc,
  title={SDC-Net: Video prediction using spatially-displaced convolution},
  author={Reda, Fitsum A and Liu, Guilin and Shih, Kevin J and Kirby, Robert and Barker, Jon and Tarjan, David and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={718--733},
  year={2018}
}

Copyright (C) 2019 NVIDIA Corporation. Yi Zhu, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao and Bryan Catanzaro.
All rights reserved. 
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
