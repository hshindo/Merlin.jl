function ∇concat!(gy::CuArray, dim::Int, xs::Var...)
    offset = 0
    for x in xs
        if !isvoid(x.grad)
            ∇concat_shiftcopy!(gy, offset, dim, x.grad)
        end
        offset += size(x, dim)
    end
end

@generated function ∇concat_shiftcopy!(gy::CuArray{T,N}, offset::Int, dim::Int, gx::CuArray{T,N}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void concat(Array<$Ct,$N> y, int offsetY, int dim, Array<$Ct,$N> x) {
        int idxX = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxX >= x.length()) return;

        int ndIdx[$N];
        x.idx2ndIdx(ndIdx, idxX);
        ndIdx[dim] += offsetY;
        x[idxX] += y(ndIdx);
    }""")
    quote
        gdims, bdims = cudims(length(gx))
        culaunch($f, gdims, bdims, gy, offset, dim-1, gx)
    end
end
