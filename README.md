<div align="center">
<h1>测试 BotVIO: A Lightweight Transformer-Based Visual-Inertial Odometry for Robotics</h1>
</div>


## 配置测试

```sh
git clone https://github.com/R-C-Group/BotVIO.git  --recursive

conda create -n botvio python=3.10
conda activate botvio

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install timm==0.4.12
```

Data Preparation:采用KITTI数据集

```sh
cd data
source data_prep.sh 
```