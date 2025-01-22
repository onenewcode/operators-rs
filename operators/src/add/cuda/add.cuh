template<class Tdata>
static __device__ void add(
    Tdata *__restrict__ c,
    int const *__restrict__ strides_c,
    Tdata const *__restrict__ a,
    int const *__restrict__ strides_a,
    Tdata const *__restrict__ b,
    int const *__restrict__ strides_b,
    int const *__restrict__ strides_i,
    int const i_strides_size) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset_c = 0;
    int offset_a = 0;
    int offset_b = 0;
#pragma unroll
    for (size_t tmp_i = 0; tmp_i < i_strides_size; ++tmp_i) {
        int k = i / strides_i[tmp_i];
        offset_c += k * strides_c[tmp_i];
        offset_a += k * strides_a[tmp_i];
        offset_b += k * strides_b[tmp_i];
        i %= strides_i[tmp_i];
    }
    c[offset_c] = a[offset_a] + b[offset_b];
}
