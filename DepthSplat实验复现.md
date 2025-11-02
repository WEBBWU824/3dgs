# DepthSplat实验复现
实验网站 https://github.com/cvg/depthsplat
这次实验复现基于 AutoDL 与 VSCode 远程连接，用linux终端实现。
## 1. 环境配置
按照官方文档，环境为 PyTorch 2.4.0、CUDA 12.4、Python 3.10，使用conda设置虚拟环境：
进入下载好的的depthsplat文件夹中，指令依次输入
```
conda create -y -n depthsplat python=3.10
```
```
conda activate depthsplat
```
```
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
```
```
pip install -r requirements.txt
```
## 2. 准备数据集
我选择用 re10k 作为数据集，按照官方文档也可以用 DL3DV 。
具体步骤如下：
1. 打开 DATASETS.md，在 RealEstate10K 中点击 [pixelSplat repo](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) ，在本地下载 re10k_subset.zip
2. 通过 AutoDL 的 notebook 在 depthsplat 文件夹底下创建一个 datasets 文件夹，用于存放输入数据，把下载好的 re10k_subset.zip 上传到datasets 文件夹里
3. 在 VSCode 中输入指令解压 `unzip re10k_subset.zip`得到 re10k_subset 文件夹，文件夹改名为 re10k
## 3.下载预训练的模型
打开[Hugging Face 🤗](https://huggingface.co/haofeixu/depthsplat) ，其中的每个模型都能在 MODEL_ZOO.md 中找到对应训练方法、训练数据等。

进入 depthsplat 文件夹，创建 checkpoints 文件夹，用于存储下载好的模型。

选择数据对应的预训练模型，这里我选择用 re10k ；打开 depthsplat/config/experiment/re10k.yaml，可以看到其中这样的代码：
```yaml
wandb:
  name: re10k
  tags: [re10k, 256x256] #训练时图片分辨率为256x256
```
所以选择用256x256的模型；根据自己的需求选择下载 large/base/small 的模型。
## 4. 训练
需要先安装 ffmeg ，输入指令
```
apt update
apt install -y ffmpeg
```
然后输入以下指令即可开始训练
```
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=checkpoints/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
mode=test \
test.save_image=true \
test.save_video=true \
test.save_gt_image=true \
test.save_input_images=true \
test.compute_scores=true \
output_dir=outputs/re10k_full_test
```
这里对于不同的数据和不同的模型需要自行修改参数，具体在 Useful configs 中查看对应功能。
## 5. 结果
训练完成后，可以看到在新建的 depthsplat 底下有个 output 文件夹，结构大致是这样
```
├── images
│   ├── 1214f2a11a9fc1ed
│   ├──	...
│   └── ffa95c3b40609c76
├── metrics
│   ├── benchmark.json
│   ├── peak_memory.json
│   ├── scores_all_avg.json
│   ├── scores_lpips_all.json
│   ├── scores_psnr_all.json
│   └── scores_ssim_all.json
└── videos
    ├── 1214f2a11a9fc1ed_frame_0_135.mp4
    ├──	...
    └── ffa95c3b40609c76_frame_0_135.mp4

```
其中videos就是渲染出的图片，与官方文档给出的一致。
<video width="256" height="256" controls>
    <source src=imgs/656381bea665bf3d_frame_0_135.mp4 type="video/mp4">
</video>
