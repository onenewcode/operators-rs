#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
template <int THREADDIMX, int BLOCKDIMX, typename Tdata, typename Tidx>
static __device__ void layer_norm(
    Tdata *__restrict__ y,
    Tdata const *__restrict__ x,
    Tidx const *__restrict__ scale,
    Tidx const *__restrict__ bias,
    float epsilon,
    int const d,
    int const nsy,
    int const dsy,
    int const nsx,
    int const dsx,
    int const dss,
    int const dsb)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    typedef cub::BlockReduce<Tdata, THREADDIMX> BlockReduce; //<float,..>里面的float表示返回值的类型
    __shared__ typename BlockReduce::TempStorage temp_sum;
    __shared__ typename BlockReduce::TempStorage temp_sum2;
    auto tmp = x[blockIdx.y * nsx + idx * dsx];

    Tdata block_sum = BlockReduce(temp_sum).Reduce(tmp, cub::Sum());
    Tdata block_sum2 = BlockReduce(temp_sum2).Reduce(tmp * tmp, cub::Sum());
    if (threadIdx.x == 0)
    {
        y[blockIdx.y * gridDim.x + blockIdx.x] = block_sum;
        y[blockIdx.y * gridDim.x + blockIdx.x + blockDim.y * blockDim.x] = block_sum2;
    }
    __syncthreads();
    // TODD 带完善第二次规约
    if (BLOCKDIMX != 1)
    {
    }
    __syncthreads();

    __shared__ Tdata cache[2];
    if (threadIdx.x == 0)
    {
        cache[0] = y[blockIdx.y * gridDim.x];
        cache[1] = y[blockIdx.y * gridDim.x+ blockDim.y * blockDim.x];
        // printf(" thread:%f %f  %f\n", cache[0],cache[1]);
        // 执行中部计算
        auto e = cache[0] / d;
        auto e2 = cache[1] / d;
        auto std = sqrtf(e2 - e * e);
        cache[0] = e;
        cache[1] = Tdata(1) / (std + Tdata(epsilon));
    }
    __syncthreads();
    
    auto tmp_x = x[blockIdx.y * nsx + idx * dsx];
    auto tmp_s = Tdata(scale[idx * dss]);
    auto tmp_b = Tdata(bias[idx * dsb]);
    y[blockIdx.y * nsy + idx * dsy] = (tmp_x - cache[0]) * cache[1] * tmp_s + tmp_b;
    printf(" thread:%d %f \n", blockIdx.y * nsy + idx * dsy,__half2float(tmp_s));
}