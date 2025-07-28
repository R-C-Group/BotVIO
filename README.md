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
pip install matplotlib
pip install gdown
```

Data Preparation:采用KITTI数据集(注意要留有空间)

```sh
cd data
source data_prep.sh 
```

然后下载预训练模型[link](https://drive.google.com/drive/folders/1D-CpdPKyOwRMFlU-sp0dhslvIBmx9oxf?usp=drive_link),并且放在`pretrain_models`路径下

~~~
cd ./pretrain_models
python download.py
~~~

计算pose的结果：

```sh
conda activate botvio
python ./evaluations/eval_odom.py
```