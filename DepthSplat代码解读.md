# DepthSplat 代码
## 整体框架梳理
depthsplat文件夹结构为
```
└─depthsplat
    ├─assets
    ├─config
    ├─scripts
    └─src
	  └─.gitignore
	  └─DATASETS.md
	  └─LICENSE
	  └─MODEL_ZOO.md
	  └─README.md
	  └─requirements.txt
```
1. `assets/`
存放数据集相关的索引、配置数据（如json文件），用于实验索引、评估时加载。
2. `config/`
存放项目运行所需的各种yaml配置文件，种类多，包括数据集、实验、损失函数、模型等相关配置。常用于训练、评估等参数自动加载。
3. `scripts/`
存放Shell脚本，用于一键启动训练、推理等任务。
4. **`src/`（核心）**
核心源码目录，包含所有模块实现
5. `DATASETS.md`
介绍支持与使用的数据集类型，数据结构说明，以及如何下载、组织和准备数据，便于配置和复现实验。
6. `MODEL_ZOO.md`
列出了项目已经训练好的模型权重下载方式、精度、适用场景和用法说明，方便复现和测试。
7. `README.md`
实验的主要流程
## 实验主流程
### 配置环境
涉及到 requirements.txt 和 config/底下的.yaml文件。
### 准备数据
数据的下载和处理说明都在`DATASETS.md`；涉及到 `src/dataset/`、`assets/`、`config/dataset/`，其中 `assets/` 底下都是`.json`文件，包含实验的索引列表和预采样代码，`src/dataset/` 包含所有读数据、采样、加载、预处理等模块。
### 训练
训练入口是`src/main.py`，通过这个文件自动调用下游的模块。
主要的代码文件还有`src/model/`：编码器、解码器模型结构与组合，包装，`src/loss/`：几类损失函数的定义和串联，`src/scripts/`：数据转换、索引生成。
### 渲染
除了`main.py`的调用，还主要用到-   `src/model/decoder/`、`src/model/model_wrapper.py`、`src/visualization/`，作用如下
|代码文件|用途  |
|--|--|
| `src/model/decoder/` |高斯渲染器，CUDA等加速|
| `src/model/model_wrapper.py` |推理统一接口封装|
| `src/visualization/` |渲染结果可视化|
## 核心代码解读
核心1：将深度和多视角特征转变为高斯参数
核心2：将核心1的转换整合到统一的训练框架中
核心3：前传：多视角输入 → 深度估计，深度+特征 → 高斯参数化，高斯渲染 → 图像重建；反传：渲染损失 → 高斯参数优化，参数梯度 → 改进深度估计，深度改进 → 更好的高斯参数
> 输入图像 → 深度估计 → 高斯参数化 → 渲染重建 → 损失反传 → 优化深度 	 
> ↑_________________________________________________|

### 核心1：深度和多视图特征到高斯的桥梁
### gaussian_adapter.py

输入：编码的高斯参数、深度图、像素坐标、相机参数
输出：四种高斯参数、缩放、旋转
```python
# 1. 数据类定义
@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]           # 3D位置 (x,y,z)
    covariances: Float[Tensor, "*batch 3 3"]   # 协方差矩阵 (3×3)
    scales: Float[Tensor, "*batch 3"]          # 缩放向量 (sx,sy,sz)
    rotations: Float[Tensor, "*batch 4"]       # 旋转四元数 (w,x,y,z)
    harmonics: Float[Tensor, "*batch 3 _"]     # 球谐系数 RGB×SH阶数
    opacities: Float[Tensor, " *batch"]        # 不透明度 α
    # *batch：* 表示任意数量的批次维度
@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float  # 高斯最小尺度
    gaussian_scale_max: float  # 高斯最大尺度
    sh_degree: int             # 球谐函数阶数
```
```python
# 2. 初始化
def __init__(self, cfg: GaussianAdapterCfg):
    super().__init__()
    self.cfg = cfg
    # 注册球谐掩码（非持久化缓冲区）
    self.register_buffer(
        "sh_mask",
        torch.ones((self.d_sh,), dtype=torch.float32),
        persistent=False,  # 不保存到checkpoint
    )
    # 为高阶球谐系数设置衰减权重
    for degree in range(1, self.cfg.sh_degree + 1):
        start_idx = degree**2
        end_idx = (degree + 1) ** 2
        self.sh_mask[start_idx:end_idx] = 0.1 * 0.25**degree
```
```python
# 3. 核心转换逻辑
def forward(
    self,
    # 相机参数
    extrinsics: Float[Tensor, "*#batch 4 4"],    # 外参矩阵 [R|t]
    intrinsics: Float[Tensor, "*#batch 3 3"] | None,  # 内参矩阵 K
    # 像素级信息
    coordinates: Float[Tensor, "*#batch 2"],     # 像素坐标 (u,v)
    depths: Float[Tensor, "*#batch"] | None,     # 深度值 d
    opacities: Float[Tensor, "*#batch"],         # 不透明度 α
    # 网络预测的原始参数
    raw_gaussians: Float[Tensor, "*#batch _"],   # 编码的高斯参数
    # 辅助信息
    image_shape: tuple[int, int],                # (H, W)
    eps: float = 1e-8,                           # 数值稳定性
    point_cloud: Float[Tensor, "*#batch 3"] | None = None,  # 未使用
    input_images: Tensor | None = None,          # 原始RGB图像
) -> Gaussians:
		# 分割raw_gaussians为三部分
		scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
		scales = torch.clamp(F.softplus(scales - 4.),# 激活函数
			min=self.cfg.gaussian_scale_min,
			max=self.cfg.gaussian_scale_max,
			)
		assert input_images is not None
		# 旋转参数归一化
		rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
		# 球谐系数处理
		sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
		# 广播到opacities形状（处理批次维度）
		sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
		if input_images is not None:
			# 重排图像：(B, V, 3, H, W) → (B, V, H*W, 1, 1, 3)
			imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
			# 更新DC分量（第0阶球谐）
			sh[..., 0] = sh[..., 0] + RGB2SH(imgs)
		# 从缩放和旋转构建协方差
		covariances = build_covariance(scales, rotations)# 函数定义在\src\model\encoder\common\gaussians.py
		# 坐标系变换，提取旋转矩阵（相机→世界）
		c2w_rotations = extrinsics[..., :3, :3]
		# 旋转协方差到世界坐标系
		covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
		# 计算3D位置
		origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)# 函数定义在\src\geometry\projection.py
		means = origins + directions * depths[..., None]
```
```python
# 4. 输出高斯参数
	return Gaussians(
		means=means,  # 世界空间3D位置
		covariances=covariances,  # 世界空间协方差
		harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),  # 旋转SH
		opacities=opacities,  # 不透明度
		scales=scales,  # 缩放向量
		rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),  # 四元数
	)
```
### gaussian.py
为3D高斯点提供几何变换的基础操作，是把深度信息转换为可渲染高斯的关键组件之一。它将高斯的缩放和旋转转换为实际的协方差矩阵，这个协方差矩阵后续会用于渲染过程。
```python
# 1. 四元数转换到旋转矩阵
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],  # 输入四元数张量
    eps: float = 1e-8,  # 数值稳定性的小量
) -> Float[Tensor, "*batch 3 3"]:  # 输出3x3旋转矩阵
    # 将四元数拆分为 i, j, k, r 四个分量
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    # 计算缩放因子，确保四元数归一化
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)
    # 构建旋转矩阵的9个元素
    # 这是标准的四元数到旋转矩阵的转换公式
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),  # R11
            two_s * (i * j - k * r),      # R12
            two_s * (i * k + j * r),      # R13
            two_s * (i * j + k * r),      # R21
            1 - two_s * (i * i + k * k),  # R22
            two_s * (j * k - i * r),      # R23
            two_s * (i * k - j * r),      # R31
            two_s * (j * k + i * r),      # R32
            1 - two_s * (i * i + j * j),  # R33
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)
```
```python
# 2. 构建协方差矩阵
def build_covariance(
    scale: Float[Tensor, "*#batch 3"],          # 3D缩放向量
    rotation_xyzw: Float[Tensor, "*#batch 4"],   # 旋转四元数
) -> Float[Tensor, "*batch 3 3"]:               # 输出3x3协方差矩阵
    # 将缩放向量转换为对角矩阵
    scale = scale.diag_embed()
    # 将四元数转换为旋转矩阵
    rotation = quaternion_to_matrix(rotation_xyzw)
    # 构建协方差矩阵：R * S * S^T * R^T
    return (
        rotation                                         # R
        @ scale                                         # S
        @ rearrange(scale, "... i j -> ... j i")       # S^T
        @ rearrange(rotation, "... i j -> ... j i")     # R^T
    )
```
### sampler.py
这个采样器是连接深度估计和高斯生成的重要桥梁，它决定了哪些点会被转换为3D高斯，直接影响最终的渲染质量和效率。
```python
# 1. 选择最重要的点来生成高斯
def forward(
    self,
    probabilities: Float[Tensor, "*batch bucket"],  # 输入概率分布
    num_samples: int,                              # 采样数量
    deterministic: bool,                           # 是否确定性采样
) -> tuple[
    Int64[Tensor, "*batch 1"],    # 采样的索引
    Float[Tensor, "*batch 1"],    # 对应的概率密度
]:
    return (
        gather_discrete_topk(probabilities, num_samples)     # 确定性：选择top-k
        if deterministic
        else sample_discrete_distribution(probabilities, num_samples)  # 随机：按概率采样
    )
```
```python
# 2. 从forward()得到的点中提取对应的值
def gather(
    self,
    index: Int64[Tensor, "*batch sample"],   # 采样的索引
    target: Shaped[Tensor, "..."],           # 要采样的目标张量
) -> Shaped[Tensor, "..."]:                  # 采样结果
# 处理维度不匹配
while len(index.shape) < len(target.shape):
    index = index[..., None]          # 扩展维度
# 广播索引形状以匹配目标
broadcasted_index_shape = list(target.shape)
broadcasted_index_shape[bucket_dim] = index.shape[bucket_dim]
index = index.broadcast_to(broadcasted_index_shape)
# 执行采样
return target.gather(dim=bucket_dim, index=index)
```

### 核心2：前向传递与交互
### encoder_depthsplat.py
端到端：从**多视角图像**输入开始，通过深度预测网络得到**深度**，再通过特征提取网络得到**特征**，结合深度和特征生成**高斯参数**，最终得到**可渲染的3D高斯**。
创新点在于可以同时监督深度预测和渲染质量，并且-   深度预测和高斯渲染在同一个前向过程中完成
```python
# 1. 初始化（在前传中调用）
def __init__(self, cfg: EncoderDepthSplatCfg):
    # 1. 深度预测网络
    self.depth_predictor = MultiViewUniMatch(...)# MultiViewUniMatch() 定义在 mv_unimatch.py
    ...
    # 2. 特征上采样网络
    self.feature_upsampler = DPTHead(...)# DPTHead() 定义在 dpt_head.py
    ...
    # 3. 高斯参数适配器
    self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)# GaussianAdapter 定义在 gaussian_adapter.py
    ...
    # 4. 高斯参数回归器
    self.gaussian_regressor = nn.Sequential(...)
    ...
    # 5. 高斯参数头部网络
    self.gaussian_head = nn.Sequential(...)
```
```python
# 2. 前向传播
def forward(self, context, ...):
    # 1. 深度预测
    results_dict = self.depth_predictor(
        context["image"],
        min_depth=1. / context["far"],
        max_depth=1. / context["near"],
        intrinsics=context["intrinsics"],
        extrinsics=context["extrinsics"],
    )
    depth = depth_preds[-1]  # [B, V, H, W]
    # 2. 特征提取和上采样
    features = self.feature_upsampler(
        results_dict["features_mono_intermediate"],
        cnn_features=results_dict["features_cnn_all_scales"],
        mv_features=results_dict["features_mv"]
    )
    # 3. 生成高斯参数
    # 合并图像、深度、匹配概率和特征
    concat = torch.cat((
        context["image"],
        depth,
        match_prob,
        features,
    ), dim=1)
    # 4. 高斯参数回归
    out = self.gaussian_regressor(concat)
    gaussians = self.gaussian_head(out)
    # 5. 转换为3D高斯
    gaussians = self.gaussian_adapter.forward(
        context["extrinsics"],
        context["intrinsics"],
        xy_ray,  # 采样点
        depths,  # 深度值
        opacities,  # 不透明度
        gaussians,  # 高斯参数
        (h, w),   # 图像尺寸
    )
```
### model_wrapper.py（1128行）
是全流程“总控和胶水”模块，基于 PyTorch Lightning 封装训练、验证、测试，串起 Encoder、Decoder、损失、日志与可视化。
把“深度↔高斯渲染”的核心前向、损失与评测全部粘合在一起，负责“怎么跑起来、怎么记录、怎么产出结果”，是实验运行层面的中枢。

### loss.py
创建一个父类，能在具体的损失函数中被继承，方便输出。
```python
# 1. 配置 Loss 类
class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
	cfg: T_cfg # 实际的配置对象
	name: str # 损失函数的名称
```
```python
# 2. 配置提取，自动匹配
def __init__(self, cfg: T_wrapper) -> None:
    super().__init__()
    # 使用包装器模式来管理配置，自动从包装器中提取实际配置并作为损失函数名称
    (field,) = fields(type(cfg)) # 使用 dataclasses.fields 获取包装器的字段
    self.cfg = getattr(cfg, field.name)    # 获取配置对象
    self.name = field.name # 使用字段名作为损失函数名称
```
```python
# 3. 抽象前向接口
@abstractmethod
def forward(
    self,
    prediction: DecoderOutput,      # decoder的输出（color和depth）
    batch: BatchedExample,          # 数据批次（包含ground truth等）
    gaussians: Gaussians,           # 生成的3D高斯
    global_step: int,              # 全局训练步数
) -> Float[Tensor, ""]:            # 返回标量损失值
    pass
```
具体的损失函数定义在`loss_mes.py`、`loss_lpips.py`、`loss/__init__py`，损失函数被调用在`model_wrapper.py`
### 核心3：面向训练的渲染
### decoder_splatting_cuda.py
负责数据适配，处理后的数据可以用于渲染，方法有：维度重组与扩展、多视角批处理包装、背景色管理、深度模式转发
```python
# 1. 类的定义与配置
@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]  # 配置类，仅包含名称标识
class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]  # 背景色张量
```
```python
# 2. 初始化
def __init__(
    self,
    cfg: DecoderSplattingCUDACfg,
    dataset_cfg: DatasetCfg,
) -> None:
    super().__init__(cfg, dataset_cfg)
    # 注册背景色为缓冲区（不参与反向传播）
    self.register_buffer(
        "background_color",
        torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
        persistent=False,
    )
```
```python
# 3. 前传（核心渲染）
def forward(
    self,
    gaussians: Gaussians, # 高斯参数
    extrinsics: Float[Tensor, "batch view 4 4"],    # 相机外参
    intrinsics: Float[Tensor, "batch view 3 3"],    # 相机内参
    near: Float[Tensor, "batch view"],  # 近平面
    far: Float[Tensor, "batch view"],   # 远平面
    image_shape: tuple[int, int],                   # 输出图像尺寸
    depth_mode: DepthRenderingMode | None = None,   # 深度渲染模式
) -> DecoderOutput:
```
```python
# 4. 渲染流程
color = render_cuda(
    rearrange(extrinsics, "b v i j -> (b v) i j"),     # 展平batch和view维度
    rearrange(intrinsics, "b v i j -> (b v) i j"),
    rearrange(near, "b v -> (b v)"),
    rearrange(far, "b v -> (b v)"),
    image_shape,
    repeat(self.background_color, "c -> (b v) c", b=b, v=v),  # 重复背景色
    # 重复高斯参数以匹配视图数
    repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),           # 位置
    repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),    # 协方差
    repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),# 球谐系数
    repeat(gaussians.opacities, "b g -> (b v) g", v=v),              # 不透明度
)
# 重排输出张量维度 (b v) c h w -> b v c h w`
```
```python
# 5. 深度渲染: 
#(1) 同样展平批次和视图维度
#(2) 调用 CUDA 深度渲染核心
#(3) 重排输出张量维度 (b v) h w -> b v h w
def render_depth(
    self,
    gaussians: Gaussians,
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    near: Float[Tensor, "batch view"],
    far: Float[Tensor, "batch view"],
    image_shape: tuple[int, int],
    mode: DepthRenderingMode = "depth",    # 深度渲染模式
) -> Float[Tensor, "batch view height width"]:
	b, v, _, _ =extrinsics.shape
	result = render_depth_cuda(
	    rearrange(extrinsics, "b v i j -> (b v) i j"),
	    rearrange(intrinsics, "b v i j -> (b v) i j"),
	    rearrange(near, "b v -> (b v)"),
	    rearrange(far, "b v -> (b v)"),
	    image_shape,
	    repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
	    repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
	    repeat(gaussians.opacities, "b g -> (b v) g", v=v),
	    mode=mode,
	)
	return rearrange(result,"(b v) h w -> b v h w", b=b, v=v)
```
### cuda_splatting.py
封装了一系列函数：投影变换、SH 系数处理、CUDA 光栅化调用、深度映射。
```python
# 1. 生成投影矩阵
def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    "映射视锥体中的点到 X/Y轴的(-1,1)范围和Z轴的(0,1)范围"
    # 计算视锥体参数
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()
    
    # 计算视锥体边界
    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right
    
    # 构建投影矩阵（注意：Z轴与OpenGL不同，范围是0-1而不是-1到1）
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)    # X缩放
    result[:, 1, 1] = 2 * near / (top - bottom)    # Y缩放
    result[:, 0, 2] = (right + left) / (right - left)  # X偏移
    result[:, 1, 2] = (top + bottom) / (top - bottom)  # Y偏移
    result[:, 3, 2] = 1                            # 透视除法标志
    result[:, 2, 2] = far / (far - near)          # Z映射系数
    result[:, 2, 3] = -(far * near) / (far - near)# Z映射偏移
    return result
```
```python
# 2. 主渲染函数
def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],        # 相机外参
    intrinsics: Float[Tensor, "batch 3 3"],        # 相机内参
    near: Float[Tensor, " batch"],                 # 近平面
    far: Float[Tensor, " batch"],                  # 远平面
    image_shape: tuple[int, int],                  # 输出图像尺寸
    background_color: Float[Tensor, "batch 3"],    # 背景色
    gaussian_means: Float[Tensor, "batch gaussian 3"],          # 高斯中心
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"], # 协方差矩阵
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"], # 球谐系数
    gaussian_opacities: Float[Tensor, "batch gaussian"],       # 不透明度
    scale_invariant: bool = True,                  # 是否进行尺度不变性处理
    use_sh: bool = True,                          # 是否使用球谐光照
)-> Float[Tensor, "batch 3 height width"]:
	assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
# (1) 尺度不变
	if scale_invariant:
	    scale = 1 / near
	    extrinsics = extrinsics.clone()
	    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]      # 缩放相机位置
	    gaussian_covariances = gaussian_covariances * scale[:, None, None, None] ** 2  # 缩放协方差
	    gaussian_means = gaussian_means * scale[:, None, None]         # 缩放高斯中心
		near = near * scale
		far = far * scale
		...
# (2) 准备渲染参数
	# 计算视场角
	fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
	tan_fov_x = (0.5 * fov_x).tan()
	tan_fov_y = (0.5 * fov_y).tan()
	# 构建投影矩阵链
	projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
	view_matrix = extrinsics.inverse()
	full_projection = view_matrix @ projection_matrix
...
# (3) 遍历每个批次，对每个批次操作：
		# 设置渲染器参数
		settings = GaussianRasterizationSettings(
		    image_height=h,
		    image_width=w,
		    tanfovx=tan_fov_x[i].item(),
		    tanfovy=tan_fov_y[i].item(),
		    bg=background_color[i],
		    viewmatrix=view_matrix[i],
		    projmatrix=full_projection[i],
		    sh_degree=degree,
		    campos=extrinsics[i, :3, 3],
		    prefiltered=False, # This matches the original usage.
			debug=False,
		)
...
		# 调用CUDA渲染器
		image, radii = rasterizer(
		    means3D=gaussian_means[i],
		    means2D=mean_gradients,
		    shs=shs[i] if use_sh else None,
		    colors_precomp=None if use_sh else shs[i, :, 0, :],
		    opacities=gaussian_opacities[i, ..., None],
		    cov3D_precomp=gaussian_covariances[i, :, row, col],
		)
```
```python
# 3. 深度渲染函数
def render_depth_cuda(
    ...,
    mode: DepthRenderingMode = "depth",  # 深度渲染模式
)-> Float[Tensor, "batch height width"]:
    # 将高斯点转换到相机空间
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), 
        "b i j, b g j -> b g i"
    )
    # 根据模式处理深度值
    fake_color = camera_space_gaussians[..., 2]  # Z值作为深度
    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()
    # 用深度值作为颜色进行渲染
    result = render_cuda(
        ...,
        repeat(fake_color, "b g -> b g c ()", c=3),  # 深度扩展为RGB通道
        gaussian_opacities,
		scale_invariant=scale_invariant,
        use_sh=False,
    )
    return result.mean(dim=1)  # 返回单通道深度图
```
### decoder.py
作用：
为不同的渲染实现提供了一个稳固的基础，同时保持了足够的灵活性来适应不同的需求。
优点：
Decoder 接口十分抽象，定义统一的输入输出接口，将渲染器抽象为处理高斯参数的黑盒；
输出灵活，通过`batch view`支持多视角渲染，并且可以配置输出分辨率。
```python
# 1. 定义类
# 深度渲染模式的字面量类型
DepthRenderingMode = Literal[
    "depth",             # 原始深度值
    "log",              # 对数深度
    "disparity",        # 视差 (1/depth)
    "relative_disparity",# 相对视差
]
# 解码器输出数据类
@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]  # 渲染的RGB图像
    depth: Float[Tensor, "batch view height width"] | None  # 可选的深度图
```
```python
# 2. 定义编码器基类
class Decoder(nn.Module, ABC, Generic[T]):
	#抽象基类，定义了高斯渲染器的接口；泛型参数T: 配置类型
    cfg: T               # 配置对象
    dataset_cfg: DatasetCfg  # 数据集配置
    def __init__(self, cfg: T, dataset_cfg: DatasetCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians,                          # 高斯参数
        extrinsics: Float[Tensor, "batch view 4 4"],   # 相机外参
        intrinsics: Float[Tensor, "batch view 3 3"],   # 相机内参
        near: Float[Tensor, "batch view"],             # 近平面
        far: Float[Tensor, "batch view"],              # 远平面
        image_shape: tuple[int, int],                  # 输出图像尺寸
        depth_mode: DepthRenderingMode | None = None,  # 深度渲染模式
    ) -> DecoderOutput:
        pass
```
## 总结
上面总共列出九个核心代码文件，他们可以分成实验的三个核心模块

> 1. 高斯参数生成模块 gaussian_adapter.py　　　#几何计算 
> ├── gaussians.py　　　　　　　　　　　　# 提供高斯几何变换的基础操作 
> └── sampler.py　　　　　　　　　　　　　# 提供对高斯点的采样策略
> 2. 训练框架 	model_wrapper.py　　　　　　　　# **整体训练流程控制** 
> ├── encoder_depthsplat.py　　　　　　　　# 深度估计和特征提取 
> └── loss.py           　　  　　　　　　　　　　　　 # 损失函数定义和调用
> 3. 渲染模块 decoder.py　　　　　　　　　　　# 定义渲染器的抽象接口 
> ├── decoder_splatting_cuda.py　　　　　　# 实现具体的CUDA渲染器 
> └── cuda_splatting.py　　　　　　　　　　# 底层CUDA渲染操作

数据在每个文件的输入输出流向如下
前传：
> 输入图像    
> 　→ encoder_depthsplat.py (深度估计+特征提取)
>  　　→ gaussian_adapter.py (转换为高斯)
>　　　→ gaussians.py (几何变换)
>　　　　→ sampler.py (点采样)
> 　　　　　→ decoder_splatting_cuda.py (渲染)
>　　　　　　→ cuda_splatting.py (CUDA操作)
>　　　　　　　→ 输出图像/深度

训练控制：

> model_wrapper.py 
> ├── 调用 encoder_depthsplat.py 生成高斯 
> ├── 调用	decoder_splatting_cuda.py 渲染图像 
> ├── 调用 loss.py 计算损失
>  └── 执行优化步骤
