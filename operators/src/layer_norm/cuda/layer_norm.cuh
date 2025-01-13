template<class Tdata>
static __device__ void gelu(
    Tdata *__restrict__ data){
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto x = float(data[i]);
    data[i] = Tdata(0.5f * x * (1.0f + erf(x / sqrtf(2.0f))));
}

