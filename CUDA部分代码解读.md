参考链接https://zhuanlan.zhihu.com/p/12181085672?share_code=di7XDXTmUXu0&utm_psn=1961015875034387522

先整体阅读链接里的目录，可以分出五类代码：

 ## 1. CUDA前向传播代码
 作用：
&emsp;&emsp;**前向传播**是把场景中每个 Gaussian 的参数经过一系列变换、投影和光度计算，最后在图像平面上合成像素颜色与深度。其目的是生成渲染结果。
&emsp;&emsp;可以输入Gaussian参数、相机参数、图像尺寸，输出渲染后的颜色图、深度、2D坐标、半径等等。

具体函数：
1、resizeFunctional()
```cpp
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}
```
用于动态调整PyTorch tensor的大小并返回其底层数据指针。CUDA代码需要动态分配缓冲区大小通过这个回调函数，CUDA代码可以请求特定大小的内存，让PyTorch tensor自动管理内存生命周期，避免手动分配或释放。

2、RasterizeGaussiansCUDA()
```cpp
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,      // 缩放比例调整参数
	const torch::Tensor& cov3D_precomp, // 预计算的 3D 协方差矩阵（如果存在）
	const torch::Tensor& viewmatrix, // 相机外参
	const torch::Tensor& projmatrix, // 投影矩阵 将世界坐标转换为剪裁坐标
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos, // 相机位置（中心点）
	const bool prefiltered, // 是否过滤掉不在视锥内的高斯点
	const bool antialiasing, // 是否使用抗锯齿
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
//////////////////////////////////////1、输入验证与初始化///////////////////////////////////////
  // 检查输入张量的维度
  
  const int P = means3D.size(0); // 高斯点数量
  const int H = image_height;
  const int W = image_width;

   // 定义整数和浮点张量的选项 用于后续张量创建。
  // .options()函数返回张量的信息 包括数据类型和设备
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);
/////////////////////////////////////////2、输出张量准备///////////////////////////////////////
  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  // 初始化一个填充为 0 的三通道图像 用于存储渲染结果。
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();
  // .contiguous()确保 Tensor 的数据在内存中是连续排列的。
  // 初始化一个填充为 0 的张量 用于存储深度值的逆。
  // 这里我并没有看懂为什么要先创建一个空张量和空指针再做替换

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  // 初始化一个填充为 0 的张量 用于存储每个高斯点的屏幕半径。
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
/////////////////////////////////////3、动态缓冲区设置/////////////////////////////////////////
  // 分配空的动态缓冲区 用于存储几何数据（geomBuffer） 排序结果（binningBuffer） 渲染图像的中间状态（imgBuffer）。
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  // 生成函数指针 后续可以用来动态调整缓冲区的大小，例如：
  // size_t chunk_size = required<GeometryState>(P);
  // char* chunkptr = geomFunc(chunk_size);
  // 调用 geomFunc(chunk_size)动态调整 geomBuffer 的大小为 chunk_size 返回调整后 geomBuffer 的数据指针。
  
  int rendered = 0; // 表示高斯点的总投影渲染次数 后续会讲到
  if(P != 0) //没有高斯点则跳过渲染
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);// 球谐系数的数量
      }
/////////////////////////////////4、调用CUDA核心渲染器///////////////////////////////////////
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_invdepthptr,
		antialiasing,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}
```
是整个3DGS渲染的Python-CUDA接口层，真正的渲染逻辑在 `CudaRasterizer::Rasterizer::forward()` 中实现。
流程：
输入: 3D高斯参数 + 相机参数 
1.预处理: 投影到2D，计算2D协方差 (geomBuffer) 
2. Tile划分: 将屏幕分成16×16的tile 
3. 排序: 按tile-depth排序高斯点 (binningBuffer)
4. α-blending: 每个tile独立渲染 (imgBuffer)
输出: RGB图像 + 深度图 + 半径信息


3、CudaRasterizer::Rasterizer::forward()
```cpp
int CudaRasterizer::Rasterizer::forward(
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const int P, int D, int M,
    const float* background,
    const int width, int height,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* opacities,
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx, float tan_fovy,
    const bool prefiltered,
    float* out_color,
    int* radii,
    bool debug)
{
/////////////////////////////////1、初始化和预处理////////////////////////////////////////////
    const float focal_y = height / (2.0f * tan_fovy);   
    const float focal_x = width / (2.0f * tan_fovx);
    // 垂直和水平方向的焦距    

    size_t chunk_size = required<GeometryState>(P);     
    // 计算存储 P 个高斯点的几何数据所需的内存大小。
    char* chunkptr = geometryBuffer(chunk_size);        
    // 调用 resizeFunctional 返回的函数对象 为几何数据分配 chunk_size 字节的内存。
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);  
    // 初始化一个 GeometryState 实例 用于存储几何数据。

    if (radii == nullptr)
    {
        radii = geomState.internal_radii;               
    }
    // 如果 radii 参数为空 使用 geomState.internal_radii 作为默认存储。

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    // 定义网格和线程块大小

    size_t img_chunk_size = required<ImageState>(width * height);               
    // 计算存储图像每个像素的中间结果（如透明度、贡献）的内存需求。
    char* img_chunkptr = imageBuffer(img_chunk_size);                           
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);  
    // 调用 imageBuffer 分配内存 并初始化 ImageState。

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
    {
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
    }
    // 检查颜色
/////////////////////////////////////2、前缀和扫描（累积偏移量）/////////////////////////////
    // 预处理高斯点，计算投影后的协方差矩阵，半径和颜色
    CHECK_CUDA(FORWARD::preprocess(
        P, D, M,                      // 高斯点数量、球谐函数阶数、球谐系数数量
        means3D,                      // 高斯点的 3D 坐标
        (glm::vec3*)scales,           // 每个高斯点的缩放比例
        scale_modifier,               // 全局缩放因子
        (glm::vec4*)rotations,        // 高斯点旋转四元数
        opacities,                    // 高斯点的透明度
        shs,                          // 球谐函数系数
        geomState.clamped,            // 是否需要裁剪
        cov3D_precomp,                // 预计算的 3D 协方差矩阵
        colors_precomp,               // 预计算的颜色
        viewmatrix,                   // 相机外参
        projmatrix,                   // 透视矩阵，世界坐标系转换为剪裁坐标系（立方体）
        (glm::vec3*)cam_pos,          // 相机位置
        width, height,                // 图像宽高
        focal_x, focal_y,             // 水平和垂直方向焦距
        tan_fovx, tan_fovy,           // 水平和垂直视场角的正切值
        radii,                        // 每个高斯点的屏幕投影半径
        geomState.means2D,            // 高斯点的 2D 投影坐标
        geomState.depths,             // 高斯点的深度
        geomState.cov3D,              // 高斯点的协方差矩阵
        geomState.rgb,                // 高斯点的颜色
        geomState.conic_opacity,      // 协方差矩阵的逆矩阵和透明度
        tile_grid,                    // CUDA 网格布局
        geomState.tiles_touched,      // 每个高斯点覆盖的瓦片数
        prefiltered                   // 是否提前过滤视锥外的高斯点
    ), debug);

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(
        geomState.scanning_space,  // 临时缓冲区，存储中间结果
        geomState.scan_size,       // 临时缓冲区的大小
        geomState.tiles_touched,   // 输入数组，每个高斯点覆盖的瓦片数量
        geomState.point_offsets,   // 输出数组，每个高斯点在瓦片列表中的偏移量
        P), debug)                 // 高斯点的总数
    // 例如，geomState.tiles_touched = [3, 2, 0, 4] 表示第一个高斯点覆盖 3 个瓦片
    // 第二个覆盖 2 个 第三个不覆盖 第四个覆盖 4 个
    // 根据前缀和计算geomState.point_offsets = [3, 5, 5, 9]

    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    // geomState.point_offsets + P - 1 指向 geomState.point_offsets 数组的最后一个元素，
    // 表示总投影渲染次数 即之前例子中的9
    // 因为 geomState.point_offsets 是一个指向数组第一个元素的指针
    // 所以 geomState.point_offsets + P - 1 是指向数组的最后一个元素的指针
    // 通过 cudaMemcpy 将值拷贝到CPU的变量 num_rendered 上

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);
    // 计算总的投影渲染次数需要的内存空间 并初始化数据结构 binningState
    // 为后续的渲染做准备
////////////////////////////////3、生成键值对（展开为tile-gaussian pairs）/////////////////////
    duplicateWithKeys << <(P + 255) / 256, 256 >> > (
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        tile_grid)
    CHECK_CUDA(, debug)
    // 将每个3D高斯投影的瓦片 tile index 和深度存到 point_list_keys_unsorted 中
    // 以键值对的形式 即 [tile index | depth]
    // 键值对表示为64位无符号整数 uint64_t [高 32 位：tile index|低 32 位：depth]
    // 将每个3D高斯的index（第几个3D gaussian）存到point_list_unsorted中

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);
    // 找到瓦片数量的最高有效位（MSB）的下一个最高位
    // 例如有 tile_grid.x * tile_grid.y = 2040 个瓦片
    // 其二进制表示为 11111111000 MSB是第 10 位 函数返回11
//////////////////////////////////4、基数排序（按tile和深度排序）//////////////////////////////
    // 对键值对列表进行排序
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space, //排序操作所需的临时存储空间和其大小
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 0, 32 + bit), debug)
    // num_rendered 是要排序的键值对总数
    // 32+bit 是因为 tile index 是键值对的高32位
    // 这里的排序将同一瓦片所对应的键值对聚集在一起，之前的键值对是根据3D高斯 index 排序的
    // 即 binningState.point_list_keys 中同一瓦片对应的键值对是连续的并按深度排序
    // binningState.point_list 是排序后其对应的高斯点的 index列表

    // 将imgState.ranges 数组中的所有元素设置为 0
    CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // 找到每个瓦片在排序后在键值对列表中的范围 即每个瓦片的起始索引和结束索引
    // 储存到imgState.ranges中
//////////////////////////////////////5、识别Tile范围////////////////////////////////////////
    if (num_rendered > 0)
        identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);
    CHECK_CUDA(, debug)
///////////////////////////////////////6、渲染（Alpha Blending）/////////////////////////////
    // 遍历每个瓦片并渲染其中的高斯点 将结果写入 out_color
    const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::render(
        tile_grid,               // 网格配置（瓦片网格的尺寸）
        block,                   // 每个线程块的线程数量
        imgState.ranges,         // 每个瓦片中高斯点的起始和结束索引（范围数组）
        binningState.point_list, // 排序后的高斯点索引列表
        width, height,           // 图像的宽度和高度
        geomState.means2D,       // 每个高斯点在屏幕上的 2D 坐标
        feature_ptr,             // 每个高斯点的颜色
        geomState.conic_opacity, // 每个高斯点的协方差逆矩阵和不透明度（opacity）
        imgState.accum_alpha,    // 每个像素累积的 alpha（透明度值）
        imgState.n_contrib,      // 每个像素的最后一个高斯点贡献信息
        background,              // 背景颜色
        out_color),debug)        // 输出的图像颜色（最终渲染结果）

    return num_rendered;
}
```
作用：将3D高斯点光栅化为2D图像

4、duplicateWithKeys()
```cpp
__global__ void duplicateWithKeys(
    int P,
    const float2* points_xy,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int* radii,
    dim3 grid)
{
/////////////////////////////////1、线程索引和边界检查/////////////////////////////////////////
    auto idx = cg::this_grid().thread_rank(); // auto 自动推导变量的类型 无需显式声明
    if (idx >= P)
        return;
   // 每个线程处理一个高斯点 通过 thread_rank() 获取当前线程的全局索引idx 
   // 如果线程的索引大于高斯点总数 P 则该线程退出
/////////////////////////////////2、获取起始偏移量////////////////////////////////////////////
    if (radii[idx] > 0)
    {
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1]; // 获取当前高斯点对应的未排序键值对列表的起始索引
////////////////////////////////3、计算覆盖的Tile范围/////////////////////////////////////////
        uint2 rect_min, rect_max;

        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
        // 使用 getRect 函数计算当前高斯点覆盖的瓦片范围 
        // getRect 计算并返回该点影响的瓦片区域的最小值和最大值。
/////////////////////////////////////////4、生成键值对（核心）/////////////////////////////////
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                uint64_t key = y * grid.x + x;
                key <<= 32;  //将tile index移到高32位
                key |= *((uint32_t*)&depths[idx]);  //将depth放到低32位
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}
```
将高斯点展开为tile-gaussian键值对，在3、里面已经调用过了。具体方法是将每个3D高斯"复制"到它在屏幕上覆盖的所有tiles中，生成用于后续排序的键值对列表。

5、getRect()
```cpp
__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
    rect_min = {
        min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
    };
    rect_max = {
        min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
    };//“+ BLOCK_X - 1)”是为了向上取整，rect_max不包含边界
}
```
计算高斯点在屏幕上覆盖的tile范围，具体方法是：给定高斯点的屏幕坐标和半径，计算它影响哪些tiles，返回tile坐标的矩形范围。
p.s. 
__ forceinline __ ：强制内联函数 
__device __：仅仅在设备(Device)端能够使用的函数，适用于开发一个明确的单任务被多次运行的工程

6、getHigherMsb()
```cpp
uint32_t getHigherMsb(uint32_t n)
{
    // 初始化最高位为中间值
    uint32_t msb = sizeof(n) * 4; 
    // n是uint32_t类型 sizeof(n) = 4字节 sizeof(n) * 4=16位
    // 一个字节有8位 一个 uint32_t（4字节）有32位

   // 二分查找
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}
```
作用：返回存储 `n` 所需的最小比特数，即 `floor(log2(n)) + 1`。用于确定基数排序需要处理的位数，确保排序只处理必要的位数，提升整体渲染性能

7、identifyTileRanges()
