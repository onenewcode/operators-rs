#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
template <int THREADDIMX, int UNIT, typename Tdata, typename Tidx>
static __device__ void layer_norm(
    Tdata *__restrict__ y,
    Tdata const *__restrict__ x,
    Tidx const *__restrict__ scale,
    Tidx const *__restrict__ bias,
    float epsilon,
    int const n,
    int const d,
    int const nsy,
    int const dsy,
    int const nsx,
    int const dsx,
    int const dss,
    int const dsb)
{
    int idx = threadIdx.x*UNIT;
    Tdata tmp1=Tdata(0);
    Tdata tmp2=Tdata(0);
    typedef cub::BlockReduce<Tdata, THREADDIMX> BlockReduce; //<float,..>里面的float表示返回值的类型
    __shared__ typename BlockReduce::TempStorage temp_sum;
    __shared__ typename BlockReduce::TempStorage temp_sum2;
    for (size_t i = idx; i < idx+UNIT; i++)
    {
        Tdata tmp_x=x[blockIdx.x * nsx + i * dsx];
        tmp1+=tmp_x;
        tmp2+=tmp_x*tmp_x;
    }
    
    Tdata block_sum = BlockReduce(temp_sum).Reduce(tmp1, cub::Sum());
    Tdata block_sum2 = BlockReduce(temp_sum2).Reduce(tmp2, cub::Sum());
    if (threadIdx.x == 0)
    {
        y[blockIdx.x] = block_sum;
        y[ blockIdx.x+ n] = block_sum2;
    }
    __syncthreads();
    __syncthreads();
    __shared__ Tdata cache[2];
    if (threadIdx.x == 0)
    {
        cache[0] = y[blockIdx.x];
        cache[1] = y[blockIdx.x+n];
        // 执行中部计算
        auto e = cache[0] / d;
        auto e2 = cache[1] / d;
        auto std = sqrtf(e2 - e * e);
        cache[0] = e;
        cache[1] = Tdata(1) / (std + epsilon);
    }
    __syncthreads();
      for (size_t i = idx; i < idx+UNIT; i++)
    {
    auto tmp_x = x[blockIdx.x * nsx + i * dsx];
    auto tmp_s = scale[i * dss];
    auto tmp_b = bias[i * dsb];
    y[blockIdx.x * nsy + i * dsy] = (tmp_x - cache[0]) * cache[1] * tmp_s + tmp_b;
    }
}