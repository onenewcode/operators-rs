#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
// assert BLOCK_SIZE >= blockDim.x
template <unsigned int BLOCK_SIZE, class Ta, class Tw>
static __device__ void padding(
    Ta *__restrict__ y_,
    int const stride_y,
    Ta const *__restrict__ x_,
    int const stride_x,
    Tw const *__restrict__ s_,
    Tw const *__restrict__ b_,
    float const epsilon)
{
    auto y = y_ + blockIdx.x * stride_y + threadIdx.x;
    float const
        x =x_[blockIdx.x * stride_x + threadIdx.x],
        s = s_[threadIdx.x],
        b = b_[threadIdx.x];
    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage sum1;
    __shared__ typename BlockOp::TempStorage sum2;
    auto average = BlockOp(sum1).Reduce(x, cub::Sum());
    auto variance = BlockOp(sum2).Reduce(x * x, cub::Sum());
    __shared__ float layer[2];
    if (threadIdx.x == 0)
    {
        layer[0] = average/ float(BLOCK_SIZE);
        layer[1] = __frcp_rn(sqrtf(float(variance) / float(BLOCK_SIZE) - powf(layer[0], 2.0)) + epsilon);
    }
    __syncthreads();

    *y = Ta((x- layer[0]) * layer[1] * s + b);
}


template <unsigned int BLOCK_SIZE, unsigned int NUM_ITEMS_THREAD, class Tw, class Ta>
static __device__ void folding(
    Ta *__restrict__ y_,
    int const stride_y,
    Ta const *__restrict__ x_,
    int const stride_x,
    Tw const *__restrict__ s_,
    Tw const *__restrict__ b_,
    float const epsilon,
    unsigned int const items_size)
{
    y_ += blockIdx.x * stride_y;
    x_ += blockIdx.x * stride_x;

    float data[NUM_ITEMS_THREAD],scale[NUM_ITEMS_THREAD],bias[NUM_ITEMS_THREAD];
    {
        using BlockOp = cub::BlockLoad<float, BLOCK_SIZE, NUM_ITEMS_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Load(x_, data, items_size, 0.f);
        BlockOp(temp_storage).Load(s_, scale, items_size, 0.f);
        BlockOp(temp_storage).Load(b_, bias, items_size, 0.f);
    }

    float sum_average = 0,sum_variance=0;
#pragma unroll
    for (unsigned int i = 0; i < NUM_ITEMS_THREAD; ++i)
    {
        sum_average+=data[i];
        sum_variance+= data[i] * data[i];
    }

    float average, variance;
    {
        using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
        __shared__ typename BlockOp::TempStorage temp_average;
        __shared__ typename BlockOp::TempStorage temp_variance;
        average = BlockOp(temp_average).Reduce(sum_average, cub::Sum());
        variance = BlockOp(temp_variance).Reduce(sum_variance, cub::Sum());
    }

     __shared__ float layer[2];
    if (threadIdx.x == 0)
    {
        layer[0] = average/ float(items_size);
        layer[1] = __frcp_rn(sqrtf(float(variance) / float(items_size) - powf(layer[0], 2.0)) + epsilon);
    }
    __syncthreads();

#pragma unroll
    for (unsigned int i = 0; i < NUM_ITEMS_THREAD; ++i)
    {
        data[i] = (data[i]- layer[0]) * layer[1] * scale[i] + bias[i];
    }

    {
        using BlockOp = cub::BlockStore<float, BLOCK_SIZE, NUM_ITEMS_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Store(y_, data, items_size);
    }
}
