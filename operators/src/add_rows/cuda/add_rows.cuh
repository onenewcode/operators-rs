template <typename Tdata, typename Tidx>
static __device__ void add_rows(
    Tdata *__restrict__ dst,
    Tdata const *__restrict__ src,
    Tidx const *__restrict__ i,
    int const b,
    int const bsd,
    int const msd,
    int const nsd,
    int const kss,
    int const nss,
    int const bsi,
    int const msi)
{
    // 目前只缓冲一层bach，减少一次乘法,用grid中的x表示bach,长度为2b，前一段存储dst，后一段存储i
    extern __shared__ int cache[];
    cache[blockIdx.x] = blockIdx.x * bsd;
    cache[blockIdx.x + b] = blockIdx.x * bsi;
    // 同步所有线程
    __syncthreads();
    auto tmp =blockIdx.z * blockDim.z + threadIdx.x;
    // 计算索引
    auto idst = cache[blockIdx.x] + blockIdx.y * msd +  tmp * nsd;
    auto isrc = i[cache[blockIdx.x + b] + blockIdx.y * msi] * kss + tmp * nss;
    dst[idst] += src[isrc];
}