# gaussian-splatting

论文网址 https://github.com/graphdeco-inria/gaussian-splatting

以下记录使用AutoDL结合VSCode完成3D Gaussian Splatting for Real-Time Radiance Field Rendering实验复现的过程，输入为论文给出的train文件夹。

## 环境配置
### AutoDL与VSCode连接
首先在AutoDL上租用一个实例，再通过VSCode的扩展Remote-SSH连接，输入登陆指令和密码并且选择Linux即可开始在VSCode上配置pytorch等环境。具体操作可见[AutoDL帮助文档](https://www.autodl.com/docs/vscode/)
### 虚拟机环境配置
进入Linux终端，首先把gaussian-splatting文件夹克隆到根目录下
```
# HTTPS
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```
然后进入gaussian-splatting文件夹。

按照官方的文档通过``conda env create --file environment.yml``和
``conda activate gaussian_splatting``即可完成环境配置，但是在输入第一个代码后，我的Linux报错，原因是官方的**environment.yml**文件适配**CUDA SDK 11**，而我租用的实例是SDK 12.1。在询问Claude后，最终的解决方法是分步安装
```
# 1. 创建纯净环境（非常快） 
conda create -n gaussian_splatting python=3.10 -y 

# 2. 激活环境 
conda activate gaussian_splatting 

# 3. 用pip安装PyTorch（比conda快很多） 
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 

# 4. 安装其他依赖 
pip install plyfile tqdm opencv-python joblib 

# 5. 编译自定义CUDA扩展 
cd submodules/diff-gaussian-rasterization 
pip install . 

cd ../simple-knn pip install . cd ../fused-ssim 
pip install .
```
此时使用
```python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"```
检查环境可以输出
```
PyTorch: 2.1.0+cu121 
CUDA: True CUDA 
version: 12.1
```
## 实验运行
### 数据集准备
根据官方文档，只需要执行它给出的一行代码
```
python train.py -s <数据集文件夹的路径>
```
即可开始训练，这里我选择用他的SfM data，点击第二个here(https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)
![输入图片说明](/imgs/2025-10-05/20MlPxLjmUK9I8L1.png)下载到自己的电脑可以得到tandt_db.zip，然后通过AutoDL的Jupyter可以把tandt_db.zip上传到云服务器（具体操作可见[AutoDL帮助文档](https://www.autodl.com/docs/jupyterlab/)），就可以通过VSCode的终端命令解压得到tandt和db两个包含数据集的文件夹。
### 运行时遇到的问题
在输入train命令后，出现报错：
```
Traceback (most recent call last): 
File "/root/gaussian_splatting/gaussian-splatting/train.py", line 13, in <module> import torch 
File "/root/miniconda3/envs/gaussian_splatting/lib/python3.10/site-packages/torch/init.py", line 235, in <module> 
 from torch._C import * # noqa: F403 
ImportError: /root/miniconda3/envs/gaussian_splatting/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```
发现PyTorch和Intel MKL库发生了冲突，需要重新安装 PyTorch。在反馈给Claude后，最后的修改方案是
```
# 1. 确保在正确环境中 
	conda activate gaussian_splatting 
# 2. 降级 NumPy 到 1.x 版本 
	pip install "numpy<2.0" 
# 3. 安装自定义 CUDA 扩展 
	cd ~/gaussian_splatting/gaussian-splatting 
# 安装 diff-gaussian-rasterization 
	cd submodules/diff-gaussian-rasterization 
	pip install . 
# 安装 simple-knn
	cd ../simple-knn 
	pip install . 
# 安装 fused-ssim（如果有） 
	cd ../fused-ssim 
	pip install . 
```
重新安装库之后再次输入train指令，发现报错simple-knn 模块找不到 PyTorch 的共享库，需要重新编译
```
# 1. 设置 LD_LIBRARY_PATH 
	export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")/lib:$LD_LIBRARY_PATH 
# 2. 清理并重新编译 simple-knn 
	cd ~/gaussian_splatting/gaussian-splatting/submodules/simple-knn 
	rm -rf build dist *.egg-info 
	pip install .
	
# 创建 conda 环境变量脚本 
	mkdir -p $CONDA_PREFIX/etc/conda/activate.d 
	cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF' 
# 添加 PyTorch 库路径 
	TORCH_LIB=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null) 
	if [ -n "$TORCH_LIB" ]; then 
		export LD_LIBRARY_PATH="$TORCH_LIB/lib:$LD_LIBRARY_PATH" 
	fi 
	EOF 
# 使脚本可执行 
	chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh 
# 重新激活环境 
	conda deactivate conda activate gaussian_splatting 
```
此时输入对于db文件夹中train文件夹的训练指令，终于可以出现Training process的进度条![输入图片说明](/imgs/2025-10-05/EOXOCiJEazJfnjum.png)
## 结果可视化

> Training process结束后可以得到云服务器上的output文件，将output文件通过终端命令压缩成zip文件，通过AutoDL的Jupyter下载到本地Windows，然后在本地进行可视化。

官方文档中给出交互式查看的方法，以下是在本地配置环境的步骤：

 1. 点击官方文档给出的here，解压后得到viewers文件夹![输入图片说明](/imgs/2025-10-06/4RCZTMa8YaTQxsrA.png)
 2. 在浏览器中打开https://developer.nvidia.com/cuda-12-1-0-download-archive，下载CUDA v12.1。
 3. 进入下载好的CUDA文件夹，找到bin文件夹中的cudart64_12.dll；打开第一步得到的viewers，打开bin文件夹；把cudart64_12.dll复制到viewers的bin文件夹。

在本地的终端进入...\output\2b9de26e-e目录下，然后输入可视化代码
```
SIBR_gaussianViewer_app.exe -m <...\output\2b9de26e-e> (这里输入绝对路径）
```
最终即可得到最终结果展现![输入图片说明](/imgs/2025-10-06/xWDBAm2B6oS5PmXd.png)
至于在SIBR的可视化界面里，一些操作按键和界面里的图形化按键可以参考官方文档，例如移动视角的按键如图里所说![输入图片说明](/imgs/2025-10-06/rU2riz3dltSZHGNF.png)
