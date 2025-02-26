<div id="top" align="center">
  
# Bot-VIO
**BotVIO: A Lightweight Transformer-Based Visual-Inertial Odometry for Robotics**
  
<a href="#license">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/>
</a>
</div>


## Table of Contents
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Download Pretrainined Models](#download-pretrainined-models)
- [Evaluation](#evaluation)
  - [Pose Results and Visualization](#pose-results)
  - [Pose Evaluation](#pose-evaluation)
  - [Depth Evaluation](#depth-evaluation)
  - [Running Time Evaluation](#running-time-evaluation)

# Setup
- Create conda environment
- Install torch==1.12.1, torchvision==0.13.1, timm==0.4.12

## Data Preparation
Please refer to [Visual-Selective-VIO](https://github.com/mingyuyng/Visual-Selective-VIO) to prepare your data. 

## Download Pretrainined Models
Please download pretrained models and place them under pretrain_models directory.

## Evaluation
### Pose Results
    python ./evaluation/eval_odom.py
    
### Pose Evaluation
    python ./evaluation/evaluate_pose_vo.py
    Please modify '--data_path' in the options.py file to your path, modify pose embedding data to float16 in PositionalEncodingFourier (depth encoder.py file), and comment out the FC layer in the pose_encoder.py
    
    python ./evaluation/evaluate_pose_vio.py
    Please modify '--data_path' in the options.py file to your path and modify pose embedding data to float16 in PositionalEncodingFourier (depth encoder.py file) 

### Depth Evaluation
    python ./evaluation/evaluate_depth.py
    Please modify '--data_path' in the options.py file to your path, and comment out the reading of IMU data in the mono_dataset.py
    
### Running Time Evaluation
    python ./evaluation/evaluate_timing.py
    Please modify '--data_path' in the options.py file to your path and modify pose embedding data to float16 in PositionalEncodingFourier (depth encoder.py file) 

### The code was implemented based on Lite-Mono.
