@generated function ∇maximum!(gy::CuArray{T}, gx::CuArray{T}, dim::Int, idx::CuArray{Cint}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void f(Array<$Ct,3> gy, Array<$Ct,3> gx, int *idx, int length) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= length) return;

        int ndIdx[3];
        gy.idx2ndIdx(ndIdx, i);
        ndIdx[1] = idx[i];
        gx(ndIdx) += gy[i];
    }
    """)
    quote
        gy3d = reshape3d(gy, dim)
        gx3d = reshape3d(gx, dim)
        gdims, bdims = cudims(length(idx))
        culaunch($f, gdims, bdims, gy3d, gx3d, idx.ptr, length(idx))
    end
end
