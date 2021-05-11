# Advanced RealTime Road Detection and Autonomous Vehicle Control using Convolutional Neural Networks and END-TO-END Driving Models. 

![](https://github.com/sieuwe1/Advanced-Real-Time-Self-Driving/blob/master/Ai_based/RealLifeDemo.gif)

## Intro
The goal of this project is to make an electric ATV drive autonomosly over normal roads using mulitple sensors like 3D camera's and RGB camera's and state of the art video processing technolgies. The electric ATV is fitted with an windshield wiper motor and costum control/feedback circuitry to control the steering and acceleration. An Arduino with Serial comminucation is used to communicate with the main computer also fitted on the ATV. This main computer which is a Jetson Nano runs the End-To-End driving model and other algortihms for driving.

## Overview
The system will have two distinct working states. 1 state is gathering traning data and the other is driving autonomous. They are explained below.

### gathering data
This state is use to gather data for the End-To-End model. To speed this process up I will be using my BMW e46 car to gather data using its can network. Also a jetson nano will be fitted with a RGB and ZED 2 Depth camera. Below is the inputs and outputs that will be saved for later training.

Inputs
RGB camera view
3D camera View
left indicator / right indicator / no indicator
IMU data?
Top view of road using street view?

Outputs
Brake angle
Throttle angle
Steering Angle
Current speed

### Driving autonomous
After the End-To-End model is trained on the data it can autonomsly drive the car. To test the performance a small electric ATV will be used. This will get sensor data from the same sensors as on the BMW and then will run interference on the model.

## TODO
V2.0 has just started so a lot needs to be done.
- Create Arduino Script for getting I-BUS and CAN-BUS data and sending this over USB to Jetson Nano.
- Create module for getting ZED 3d depth map in black white format.
- Create module for new segmentation model using jetson interference library
- Create new End-To-End driving model using DonkeyCar behavrial model and adding inputs to this model.
- Create module for training model
- Combined V1.0 code with modules above and rewrite messy parts. 
- Test and train new driving model

