template <class Tdata, class Tidx>
static __device__ void add_rows(
    Tdata *__restrict__ dst,
    Tdata const *__restrict__ src,
    Tidx const *__restrict__ i,
    int const bsd,
    int const msd,
    int const nsd,
    int const kss,
    int const nss,
    int const bsi,
    int const msi)
{
    auto idx_n = blockIdx.x * blockDim.x + threadIdx.x;
    auto idst = blockIdx.z * bsd + blockIdx.y * msd + idx_n * nsd;
    auto isrc = i[blockIdx.z * bsi + blockIdx.y * msi] * kss + idx_n * nss;
    dst[idst] += src[isrc];
}