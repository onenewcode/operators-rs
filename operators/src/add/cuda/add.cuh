template<class Tdata>
static __device__ void add(
    Tdata *__restrict__ c,
    int const *__restrict__ c_strides,
    Tdata const *__restrict__ a,
    int const *__restrict__ a_strides,
    Tdata const *__restrict__ b,
    int const *__restrict__ b_strides,
    int const count,
    int const *__restrict__ i_strides,
    int const i_strides_size) {
    // 使用一维grid
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t tmp_i = 0; tmp_i < i_strides_size; ++tmp_i) {
        int k = i / i_strides[tmp_i];
        c += k * c_strides[tmp_i];
        a += k * a_strides[tmp_i];
        b += k * b_strides[tmp_i];
        i %= i_strides[tmp_i];
    }
    *c = *a + *b;
}