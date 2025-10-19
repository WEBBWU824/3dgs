参考链接https://zhuanlan.zhihu.com/p/12181085672?share_code=di7XDXTmUXu0&utm_psn=1961015875034387522

成功完成实验后，可以看到有一个submodules文件夹，里面就是CUDA代码。

整体阅读参考链接里的目录，可以主要分为**前传**和**反传**代码：

 ## CUDA前向传播代码
 作用：
**前向传播**是把场景中每个 Gaussian 的参数经过一系列变换、投影和光度计算，最后在图像平面上合成像素颜色与深度。其目的是生成渲染结果。
可以输入Gaussian参数、相机参数、图像尺寸，输出渲染后的颜色图、深度、2D坐标、半径等等。

具体函数：
1. resizeFunctional()
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

2. RasterizeGaussiansCUDA()
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


3. CudaRasterizer::Rasterizer::forward()
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

4. duplicateWithKeys()
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

5. getRect()
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
`__forceinline__`：强制内联函数 
`__device__`：仅仅在设备(Device)端能够使用的函数，适用于开发一个明确的单任务被多次运行的工程

6. getHigherMsb()
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

7. identifyTileRanges()
```cpp
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    uint64_t key = point_list_keys[idx];
    uint32_t currtile = key >> 32; //tile index 是 key 的高32位
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}
```
在排序后的键值对列表中，找到每个tile对应的高斯点范围，存储到 ranges 数组。
p.s.
`__global__`：在GPU端调用且在GPU端执行的的函数`__global___`和`__device__`不能同时使用，在编译器中会被编译为内联函数

8. required()
```cpp
size_t required(size_t P)
{
    char* size = nullptr;
    T::fromChunk(size, P);
    return ((size_t)size) + 128;
}
```
计算存储 P 个元素的数据结构 T 所需的总内存字节数（包含128字节对齐余量）。配合fromChunk和obtain函数使用。

9. obtain()
```cpp
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}
```
从一块连续内存 chunk 中分配 count 个类型为 T 的对齐元素，并更新 chunk 指针。 
作用：自动对齐内存，模板化，高效使用。

10. GeometryState
```cpp
struct GeometryState {
    // 深度和视锥剔除
    float* depths;              // [P] 每个高斯的深度值
    bool* clamped;              // [P×3] 是否被裁剪（3个值：x,y,z方向）
    
    // 屏幕投影
    int* internal_radii;        // [P] 屏幕空间半径（像素）
    float2* means2D;            // [P] 2D投影坐标
    
    // 协方差和渲染参数
    float* cov3D;               // [P×6] 3D协方差矩阵（对称矩阵6个独立元素）
    float4* conic_opacity;      // [P] [conic.x, conic.y, conic.z, opacity]
    float* rgb;                 // [P×3] 计算后的颜色
    
    // Tile覆盖信息
    uint32_t* tiles_touched;    // [P] 每个高斯覆盖的tile数
    uint32_t* point_offsets;    // [P] 前缀和结果，全局偏移量
    
    // CUB库临时缓冲区
    char* scanning_space;       // [scan_size] 前缀和算法的临时空间
    size_t scan_size;           // scanning_space 的大小
};
// 从一个预分配的内存块 chunk 中 为 GeometryState 的各个成员变量分配所需的内存
// 如果 P（元素数量）或成员变量类型发生变化， obtain 会动态计算所需的内存，不需要手动调整。
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
    GeometryState geom;
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.clamped, P * 3, 128);
    obtain(chunk, geom.internal_radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.cov3D, P * 6, 128);
    obtain(chunk, geom.conic_opacity, P, 128);
    obtain(chunk, geom.rgb, P * 3, 128);
    obtain(chunk, geom.tiles_touched, P, 128);

    // 传入 nullptr 作为临时缓冲区指针 表示当前调用只计算排序所需的临时缓冲区大小 不实际执行计算
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}
```
存储所有3D高斯的各个参数的结构体，负责几何与光栅化准备阶段
`fromChunk`：本质是将一块连续的内存块（chunk）划分成多个有意义的数组

12. ImageState
```cpp
struct ImageState
{
    uint2* ranges;       // 每个瓦片 (tile) 中高斯点索引范围
    uint32_t* n_contrib; // 每个像素最后一个贡献的高斯点的索引
    float* accum_alpha;  // 每个像素的累积透明度（alpha 值）

    static ImageState fromChunk(char*& chunk, size_t N);
};

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
    ImageState img;
    obtain(chunk, img.accum_alpha, N, 128); // N 是图像中总像素数
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.ranges, N, 128);
    return img;
}
```
记录每个像素的渲染结果，负责最终颜色累积与透明度计算

13. BinningState
```cpp
struct BinningState
{
    size_t sorting_size;                   // 存储用于排序操作的缓冲区大小
    uint64_t* point_list_keys_unsorted;    // 未排序的键列表
    uint64_t* point_list_keys;             // 排序后的键列表
    uint32_t* point_list_unsorted;         // 未排序的点列表
    uint32_t* point_list;                  // 排序后的点列表
    char* list_sorting_space;              // 用于排序操作的缓冲区

    static BinningState fromChunk(char*& chunk, size_t P);
};

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);

    // 如果传入 nullptr 作为临时缓冲区指针 函数不会实际执行排序 而是返回排序所需的临时缓冲区大小
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted, binning.point_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}
```
将高斯点分配到屏幕 tile，负责高斯点与瓦片的排序与匹配

## CUDA前传预处理代码
1. FORWARD::preprocess()
```cpp
void FORWARD::preprocess(
    int P,                       // 高斯点的总数
    int D,                       // 球谐函数的阶数（Degree），表示使用的球谐函数的复杂度
    int M,                       // 球谐函数的系数数量，通常为 (degree+1)^2

    const float* means3D,        // 每个高斯点的 3D 均值位置（P x 3 数组，依次存储 X、Y、Z 坐标）
    const glm::vec3* scales,     // 每个高斯点的尺度参数，定义高斯点在 3D 空间中的大小
    const float scale_modifier,  // 全局尺度修正系数，用于调整高斯点的整体大小

    const glm::vec4* rotations,  // 每个高斯点的旋转四元数，定义高斯点的方向（P x 4 数组，依次存储 W, X, Y, Z）
    const float* opacities,      // 每个高斯点的不透明度（P 大小的数组）

    const float* shs,            // 每个高斯点的球谐函数系数（P x M x 3，分别为 R、G、B 通道的系数）
    bool* clamped,               // 标志位数组，用于记录颜色值是否被截断（例如，当球谐函数生成的 RGB 值小于 0 时）

    const float* cov3D_precomp,  // 预先计算好的 3D 协方差矩阵，如果为 nullptr 则需要动态计算（P x 6，存储上三角矩阵的六个元素）
    const float* colors_precomp, // 预先计算好的颜色值（P x 3 数组），如果为 nullptr 则需要通过球谐函数动态计算

    const float* viewmatrix,     // 视图矩阵（4x4 矩阵），将世界坐标转换为相机坐标
    const float* projmatrix,     // 投影矩阵（4x4 矩阵），将相机坐标转换剪裁坐标

    const glm::vec3* cam_pos,    // 摄像机的 3D 位置（X, Y, Z 坐标）

    const int W,                 // 图像的宽度（以像素为单位）
    const int H,                 // 图像的高度（以像素为单位）

    const float focal_x,         // 水平方向的焦距（根据图像宽度和水平视场角计算）
    const float focal_y,         // 垂直方向的焦距（根据图像高度和垂直视场角计算）
    const float tan_fovx,        // 水平视场角的一半的正切值，用于投影计算
    const float tan_fovy,        // 垂直视场角的一半的正切值，用于投影计算

    int* radii,                  // 每个高斯点在屏幕上的最大半径（P 大小的数组，输出）
    float2* means2D,             // 每个高斯点投影到屏幕上的 2D 均值位置（P x 2 数组，输出）
    float* depths,               // 每个高斯点的深度值（与相机的距离，P 大小的数组，输出）
    float* cov3Ds,               // 每个高斯点的 3D 协方差矩阵，如果没有预计算值，则动态计算（P x 6 数组，输出）
    float* rgb,                  // 每个高斯点的颜色（P x 3 数组，输出）
    float4* conic_opacity,       // 每个高斯点的 2D 协方差矩阵的逆矩阵和不透明度（P x 4 数组，输出）
    const dim3 grid,             // CUDA 网格维度（用于划分瓦片）
    uint32_t* tiles_touched,     // 每个高斯点影响的瓦片数量（P 大小的数组，输出）

    bool prefiltered             // 是否启用了预过滤，如果启用，直接忽略视锥外的高斯点
)
{
    preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
        P, D, M,               
        means3D,               
        scales,                
        scale_modifier,        
        rotations,             
        opacities,             
        shs,                   
        clamped,               
        cov3D_precomp,         
        colors_precomp,        
        viewmatrix,            
        projmatrix,            
        cam_pos,               
        W, H,                  
        tan_fovx, tan_fovy,    
        focal_x, focal_y,      
        radii,                 
        means2D,               
        depths,                
        cov3Ds,                
        rgb,                   
        conic_opacity,         
        grid,                  
        tiles_touched,         
        prefiltered            
        );
}
```
在渲染前，对所有 3D 高斯点进行预处理**，将它们从三维世界空间投影到屏幕空间，为后续的光栅化或 splatting 做准备。

2. preprocessCUDA()
```cpp
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
    const float* orig_points,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* radii,
    float2* points_xy_image,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered)
{
    // 获取线程索引 每个线程负责处理一个高斯点
    // 如果线程索引超出高斯点数量范围 则直接退出
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // 将当前高斯点的半径和瓦片影响范围初始化为 0
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // 判断高斯点是否在视锥内
    float3 p_view;  
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    //从输入的 orig_points(means3D) 数组中提取当前高斯点的位置
    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    // 将 3D 点转换到裁剪坐标系（就是把相机坐标系视锥拉成立方体）
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    // 到标准化设备坐标系（NDC）
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

   // 如果cov3D_precomp 不为空，则直接使用预计算的协方差矩阵
   // 如果为空，则通过 scales 和 rotations 计算高斯点的 3D 协方差矩阵
    const float* cov3D;
    if (cov3D_precomp != nullptr)
    {
        // idx * 6 是因为协方差矩阵是一个 3X3 的对称矩阵
        // 因此只需存储上三角部分的6个元素即可
        cov3D = cov3D_precomp + idx * 6;
    }
    else
    {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        cov3D = cov3Ds + idx * 6;
    }

    // 将3D空间的协方差矩阵投影到屏幕空间，得到2D的协方差矩阵
    float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    float det = (cov.x * cov.z - cov.y * cov.y); // 计算矩阵的行列式
    if (det == 0.0f)
        return;  // 如果行列式为零 矩阵是不可逆的 直接返回 跳过当前高斯点的处理
    float det_inv = 1.f / det;
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv }; 
    // 计算 2D 协方差矩阵的逆矩阵

    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det)); // 最大特征值 对应高斯分布的最长轴
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det)); // 最小特征值 对应高斯分布的最短轴
    // max(0.1f, ...) 确保开方操作不会出现负值 给特征值设置一个最小值 确保半径不会过小
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2))); // 使用 max(lambda1, lambda2) 确保稳定性
    // 由于高斯分布可能是椭圆形的 最长轴的半径由最大特征值决定
    // 计算半径时 取特征值的平方根作为高斯分布的尺度
    // 这里取 3  作为高斯点的有效影响范围 将椭圆近似为一个圆

    // 将标准化设备坐标系（NDC）转换为像素坐标
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
    uint2 rect_min, rect_max;

    // 计算当前的 2D 高斯落在哪几个tile上
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    // 如果没有命中任何一个tile则直接返回
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // 如果没有预计算的颜色，则通过球谐函数 (SH) 动态生成颜色
    if (colors_precomp == nullptr)
    {
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }

    // Store some useful helper data for the next steps.
    depths[idx] = p_view.z;
    radii[idx] = my_radius;
    points_xy_image[idx] = point_image;
    conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}
```
为每一个高斯点计算它在屏幕上的投影信息，包括位置、半径、深度、协方差矩阵（2D形状）、颜色等。
流程：

    1. 判断该点是否在视锥内
    2. 变换到相机坐标与裁剪坐标
    3. 获取或计算 3D 协方差矩阵
    4. 将协方差矩阵投影到 2D 屏幕空间
    5. 计算 2D 协方差逆矩阵（用于渲染时的椭圆形 splat）
    6. 求出最大特征值 → 得到屏幕半径（影响范围）
    7. 计算屏幕像素坐标
    8. 判断该点落在哪些 tile 上
    9. 若需要则计算颜色（球谐函数）
    10. 把结果写入输出缓冲区

3. in_frustum()
```cpp
__forceinline__ __device__ bool in_frustum(int idx,
    const float* orig_points,
    const float* viewmatrix,
    const float* projmatrix,
    bool prefiltered,
    float3& p_view)
{
    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

    // 将高斯点转换到标准化设备坐标系（NDC）
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
    // 转换到相机坐标系
    p_view = transformPoint4x3(p_orig, viewmatrix);

    // 深度小于 0.2 时，视为太靠近相机或在相机后方，直接剔除
    if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
    {
        // 如果程序设置为已经过滤 (prefiltered = true)，但当前点仍然被剔除，说明逻辑出错，直接抛出错误
        if (prefiltered)
        {
            printf("Point is filtered although prefiltered is set. This shouldn't happen!");
            __trap();
        }
        return false;
    }
    return true;
}
```
判断一个3dgs点是否在相机视锥体内，剔除靠近相机或无效的点。

4. computeCov3D()
```cpp
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    glm::mat3 S = glm::mat3(1.0f);  // 单位矩阵初始化
    S[0][0] = mod * scale.x; // 乘以统一缩放系数 mod
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    glm::vec4 q = rot;
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // 四元数转换为旋转矩阵
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;

    // 变换矩阵 M 与其转置相乘，得到协方差矩阵 Σ=M.T M
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}
```
给定一个三维点的 缩放参数 scale、旋转参数 rot（四元数表示）和一个全局缩放系数 mod，  
计算其三维协方差矩阵的上三角部分（共6个唯一分量），并存入 cov3D。（数学层面）

5. computeCov2D()
```cpp
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, 
                               float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
    // 将当前3D高斯的中心点从世界坐标系投影到相机坐标系
    float3 t = transformPoint4x3(mean, viewmatrix);

    // limx 和 limy 定义了屏幕空间的限制值 这个值是通过 FOV 计算得出的
    // 1.3f 是一个放大因子 保证能够容纳更广的场景
    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    // 通过深度值 t.z 进行归一化
    const float txtz = t.x / t.z; 
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z; // 确保了 txtz  的值在 -limx 和 limx 之间
    t.y = min(limy, max(-limy, tytz)) * t.z;

    // 投影变换的雅可比矩阵 J 
    // focal_x / t.z 和 focal_y / t.z 是在 x 和 y 方向的缩放因子
    // -(focal_x * t.x) / (t.z * t.z) 和 -(focal_y * t.y) / (t.z * t.z) 计算了投影对 x 和 y 方向的偏导数
    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    // 从 viewmatrix 中提取 3x3 的旋转矩阵部分存储到W中
    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);
    
    glm::mat3 T = W * J;

    // 协方差矩阵
    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    // Σ′=JWΣ WT JT
    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // 在协方差矩阵的对角线上加一个小的正数
    // 确保每个高斯分布至少有一个像素的宽度和高度
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}
```
将一个 3D 高斯的协方差矩阵$\sum_{3D}$经过相机投影转换为屏幕空间下的 2D 协方差矩阵 $\sum_{2D}$​，用于确定该高斯在屏幕上的椭圆形状与模糊范围。返回值 `{cov[0][0], cov[0][1], cov[1][1]}` 是二维协方差矩阵的上三角部分

6. computeColorFromSH()
```cpp
__device__ glm::vec3 computeColorFromSH(
    int idx,                // 当前高斯点的索引
    int deg,                // 球谐函数 (SH) 的最高阶数
    int max_coeffs,         // 最大的球谐系数个数
    const glm::vec3* means, // 高斯点的 3D 均值坐标 (位置)
    glm::vec3 campos,       // 相机中心位置
    const float* shs,       // 球谐系数数组，存储每个高斯点的颜色特征
    bool* clamped           // 记录 RGB 是否被截断为非负数
)
{
    glm::vec3 pos = means[idx];    // 当前高斯点的位置
    glm::vec3 dir = pos - campos;  // 从相机到高斯点的方向向量
    dir = dir / glm::length(dir);  // 单位化方向向量

    // 偏移量 idx * max_coeffs 找到当前高斯点对应的球谐系数起始位置
    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
    // 使用球谐函数 0 阶（常数项）的系数 sh[0] 初始化颜色值
    // SH_C0 是球谐函数的归一化常数 对应 0 阶项
    // 表示均匀光照或颜色偏置 类似于环境光的效果
    glm::vec3 result = SH_C0 * sh[0];

    if (deg > 0)
    {
        // 计算1阶球谐函数贡献
        // 1阶球谐函数用于表示方向相关的线性变化 比如光照方向性
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1)
        {
            // 计算2阶球谐函数贡献
            // 2 阶球谐函数可以编码更复杂的方向性变化 比如双向光照或高阶反射
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    // 为了防止颜色出现负值 对计算结果加偏移量 0.5
    // 这种偏移使得颜色值保持在正范围内 有助于视觉效果的稳定
    result += 0.5f;

    // 判断计算出来的颜色分量（result.x, result.y, result.z）是否小于 0 并记录到 clamped 数组中
    // 使用 glm::max(result, 0.0f) 将负值颜色截断为 0 确保最终输出的颜色是合法的 RGB 值
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, 0.0f);
}
```
对每个高斯点，根据相机方向，从其球谐系数中恢复颜色，得到 view-dependent color。最终在渲染阶段，每个像素的颜色由多个高斯点加权合成。

7. transformPoint
 ```cpp
__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    };
    return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
    float4 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    };
    return transformed;
}
```
 | 名称 | 变换类型 | 用途 |
 |--|--|--|
 | transformPoint4x4 | 完整齐次变换 | 常用于投影、视图空间转换 |
 | transformPoint4x3 | 仿射变换 (旋转+平移) | 常用于世界坐标变换、几何点位置 |

8. ndc2Pix()
```cpp
__forceinline__ __device__ float ndc2Pix(float v, int S) // S 是屏幕分辨率
{
    return ((v + 1.0) * S - 1.0) * 0.5;
}
```
将归一化设备坐标（Normalized Device Coordinates, NDC）[-1, 1] 转换为像素坐标（pixel coordinate）[0, S-1]

## CUDA前传渲染代码
1. FORWARD::render()
```cpp
void FORWARD::render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float* colors,
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    const float* bg_color,
    float* out_color)
{
    renderCUDA<NUM_CHANNELS> << <grid, block >> > (
        ranges,             // 每个 tile (图像小块) 的范围 表示哪些高斯点影响该 tile
        point_list,         // 按深度和 tile 排序的 高斯索引列表
        W, H,               // 图像的宽和高
        means2D,            // 每个高斯分布的2D坐标中心
        colors,             // 每个高斯点的颜色信息
        conic_opacity,      // 协方差矩阵逆矩阵和不透明度的组合
        final_T,            // 渲染过程后每个像素的最终透明度或透射率值
        n_contrib,          // 每个像素的最后贡献高斯点的编号
        bg_color,           // 背景颜色
        out_color);         // 输出图像
}
```
作用：把一批3dgs点投影到屏幕上，并在 GPU 上以 tile 并行的方式，计算出每个像素的最终颜色和透明度。函数本身不计算，只是把参数准备好并启动 kernel。

2. ***renderCUDA()
```cpp
// __global__ CUDA 核函数 运行在 GPU 设备上 不能直接由 CPU 调用 而是通过内核启动配置
// __launch_bounds__(BLOCK_X * BLOCK_Y) 性能优化提示 告诉编译器每个线程块最多包含 BLOCK_X * BLOCK_Y 个线程
// 有助于编译器进行寄存器分配优化和资源调度 避免线程资源不足导致的性能下降
// __restrict__关键字在代码后详细讲
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color)
{
    // 当前线程块 用于同步和共享数据
    auto block = cg::this_thread_block();
    // 水平方向的块数 通过宽度除以每块的像素数量计算
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; 
    // 当前处理的tile的左上角的像素坐标
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    // 当前处理的tile的右下角的像素坐标
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    // 当前处理的像素坐标
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    // 当前处理的像素id
    uint32_t pix_id = W * pix.y + pix.x;
    // 将整数像素坐标转换为浮点数 提高计算精度
    float2 pixf = { (float)pix.x, (float)pix.y };

    // 检查当前线程是否负责有效像素 或位于图像边界之外
    bool inside = pix.x < W&& pix.y < H;
    // 已完成的线程可以帮助加载数据 但不执行光栅化计算
    bool done = !inside;

    // 当前处理的 tile 对应的高斯的起始id和结束id
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    // 将高斯点分批处理 每批最多 BLOCK_SIZE 个高斯点
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // 还有多少3D gaussian需要处理
    int toDo = range.y - range.x;

    // 共享内存分配
    // 用于在一个线程块内存储加载的高斯参数 提高访问速度
    __shared__ int collected_id[BLOCK_SIZE];  // 存储当前批次高斯点的ID
    __shared__ float2 collected_xy[BLOCK_SIZE];  // 存储当前批次高斯点的中心坐标
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];  //存储协方差矩阵逆和透明度

    // Initialize helper variables
    float T = 1.0f;                  // 初始透明度
    uint32_t contributor = 0;        
    uint32_t last_contributor = 0;
    float C[CHANNELS] = { 0 };       // 初始化颜色

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // 统计已完成线程数量 如果所有线程都完成 则退出循环。
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        int progress = i * BLOCK_SIZE + block.thread_rank();
        // 如果 range.x + progress 超出 range.y 表示没有更多的高斯点可处理
        if (range.x + progress < range.y)
        {
            // 获取当前线程负责的高斯点索引
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];  // 2D 屏幕空间坐标
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id]; // 协方差矩阵逆和透明度
        }
        // 同步线程块中的所有线程 确保所有线程都完成数据加载后再继续执行
        block.sync();

        // 遍历当前批次的高斯点
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            // Keep track of current position in range
            // 用于追踪像素的贡献来源 后续计算梯度时会用到
            contributor++;

            // Resample using conic matrix (cf. "Surface 
            // Splatting" by Zwicker et al., 2001)
            float2 xy = collected_xy[j];                     // 2D 中心点坐标
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };     // 高斯点到像素的偏移量
            float4 con_o = collected_conic_opacity[j];       // 协方差矩阵逆和透明度
            // 计算高斯分布的权重 用于确定像素在光栅化过程中的贡献程度
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f)
                continue;

            // Eq. (2) from 3D Gaussian splatting paper.
            // Obtain alpha by multiplying with Gaussian opacity
            // and its exponential falloff from mean.
            // Avoid numerical instabilities (see paper appendix). 
            float alpha = min(0.99f, con_o.w * exp(power)); // con_o.w 是高斯的透明度 
            if (alpha < 1.0f / 255.0f)
                continue;
            float test_T = T * (1 - alpha);
            // 如果透射率已接近 0 表示像素已经足够不透明 不再需要继续计算
            if (test_T < 0.0001f)
            {
                // 标记线程完成任务 跳出后续循环 提高效率
                done = true;
                continue;
            }

            // Eq. (3) from 3D Gaussian splatting paper.
            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

            T = test_T;

            // Keep track of last range entry to update this
            // pixel.
            last_contributor = contributor;
        }
    }

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside)
    {
        final_T[pix_id] = T;                     // 保存最终透射率
        n_contrib[pix_id] = last_contributor;    // 记录最后贡献的高斯点索引
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
    }
}
```
实现了 按 tile 并行的 3D Gaussian Splatting 前向渲染（前向累积）：
每个线程块负责一个 tile（小块像素），块内线程并行地把这个 tile 里每个像素与该 tile 影响到的高斯点（按深度/排序）做加权合成，输出像素颜色 out_color 和最终透射率 final_T。
p.s.
`__launch_bounds__`告诉编译器内核的线程块大小和最大线程数，以便优化线程分配和资源管理。
`__restrict__`在该指针的生命周期内，其指向的对象不会被别的指针所引用。可以减少内存的访问次数，提高性能。

## CUDA反向传播代码
 1. RasterizeGaussiansBackwardCUDA()
```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R, //渲染时涉及的像素范围
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  //初始化返回的梯度张量
  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_invdepthptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dinvdepthsptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(
  dL_dmeans2D, //每个高斯在二维屏幕上的中心位置的梯度
  dL_dcolors, //颜色梯度 (P, C)
  dL_dopacity, //不透明度梯度 (P, 1)
  dL_dmeans3D, //3D 均值位置梯度 (P, 3)
  dL_dcov3D, //3D 协方差上三角 6 元素梯度 (P,6)
  dL_dsh, //球谐系数梯度 (P, M, 3)
  dL_dscales, //尺度参数梯度 (P,3)
  dL_drotations);//四元数旋转梯度 (P,4)
}
```
作用是：
      1. 为 3D Gaussian Splatting 的前向渲染计算**反向梯度**（backward）。
      2. 输入包括**渲染时的各种参数**（点位、颜色、协方差、相机矩阵、球谐系数等）以及对**输出（颜色、invdepth）的上游梯度** dL_dout_color, dL_dout_invdepth。  
      3. 它在 C++ 层分配/准备好若干**输出梯度张量**（例如 dL_dmeans3D、dL_dcolors、dL_dconic、dL_dscales 等），并把这些张量的裸指针传给底层 CUDA 后端函数 CudaRasterizer::Rasterizer::backward(...)，由 CUDA 内核直接写回这些内存。
        4. 最终把这些梯度以 std::tuple **返回给 Python / PyTorch**。
 
2. CudaRasterizer::Rasterizer::backward()
```cpp
void CudaRasterizer::Rasterizer::backward(
    const int P, int D, int M, int R,
    const float* background,
    const int width, int height,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* campos,
    const float tan_fovx, float tan_fovy,
    const int* radii,
    char* geom_buffer,
    char* binning_buffer,
    char* img_buffer,
    const float* dL_dpix,
    float* dL_dmean2D,
    float* dL_dconic,
    float* dL_dopacity,
    float* dL_dcolor,
    float* dL_dmean3D,
    float* dL_dcov3D,
    float* dL_dsh,
    float* dL_dscale,
    float* dL_drot,
    bool debug)
{
    // 从提供的缓冲区中恢复几何、分桶和图像状态
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
    BinningState binningState = BinningState::fromChunk(binning_buffer, R);
    ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

    // 如果没有提供半径信息 使用前向传播过程中计算的屏幕空间半径
    if (radii == nullptr)
    {
        radii = geomState.internal_radii;
    }

    // 根据视场角和图像大小计算焦距 用于梯度反向传播中投影变换的导数计算
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    // 设置 CUDA 网格和块大小
    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // 计算损失对像素颜色的梯度 并反向传播到高斯点的颜色、2D 均值、2D 协方差和不透明度上
    // 使用前向传播的中间结果(如 means2D, conic_opacity)加速反向传播计算
    // 更新初步梯度信息
    const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    CHECK_CUDA(BACKWARD::render(
        tile_grid,
        block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        background,
        geomState.means2D,
        geomState.conic_opacity,
        color_ptr,
        imgState.accum_alpha,
        imgState.n_contrib,
        dL_dpix,
        (float3*)dL_dmean2D,
        (float4*)dL_dconic,
        dL_dopacity,
        dL_dcolor), debug)

    // 从 2D 梯度传递到 3D 均值和协方差矩阵
    // 更新尺度和旋转参数的梯度 用于进一步优化形状和方向
    // 传播颜色梯度到球谐系数 以优化颜色表示
    const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
    CHECK_CUDA(BACKWARD::preprocess(P, D, M,
        (float3*)means3D,
        radii,
        shs,
        geomState.clamped,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        scale_modifier,
        cov3D_ptr,
        viewmatrix,
        projmatrix,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        (glm::vec3*)campos,
        (float3*)dL_dmean2D,
        dL_dconic,
        (glm::vec3*)dL_dmean3D,
        dL_dcolor,
        dL_dcov3D,
        dL_dsh,
        (glm::vec3*)dL_dscale,
        (glm::vec4*)dL_drot), debug)
}
```
对像素损失的上游梯度（dL_dpix，即对输出颜色$\alpha$ 的梯度）反向传播回高斯参数（位置、颜色、不透明度、协方差/尺度/旋转、SH 系数等）。分为两个步骤：
-   **像素级反向（render ）**（像素 → 局部属性梯度）：使用前向渲染阶段产生的中间状态（`geomState`, `binningState`, `imgState`）以及 `dL_dpix`，计算并累加每个高斯的若干中间梯度，最直接的是对 2D 屏幕中心、2D conic、不透明度、颜色的梯度。这个工作由 `BACKWARD::render(...)` 完成。
    
-   **参数级反向（preprocess ）**（屏幕属性梯度 → 3D 参数梯度）：把第二步得到的 2D 空间梯度继续往上推，转换成对 3D 均值 `means3D`、3D 协方差 `cov3D`、尺度 `scales`、旋转 `rotations`、以及 SH 系数 `shs` 的梯度。这由 `BACKWARD::preprocess(...)` 完成。

# 对比前向传播和反向传播
|  |前传 |反传 |
|--|--|--|
| 数据流向 | 3D参数 → 2D参数 → 像素颜色 |像素梯度 → 2D梯度 → 3D梯度|
| 遍历顺序 | 从前到后（深度排序） |    从后到前（反向遍历） |
| 目标 |生成最终图像  |计算参数梯度 |
|计算公式 | 主要是变换和混合 |  主要是链式法则和偏导数 |
| 并行策略 | 每个线程独立处理 | 需要原子操作处理梯度累积  |

p.s.原子操作：操作不可再分。在并发场景下，必须要用到原子操作保证数据是一致的，否则不同线程下可能结果与期望不同。



