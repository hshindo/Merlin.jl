@generated function softmax_crossentropy(p::CuVector{Cint}, logx::CuMatrix{T}) where T
    Ct = LibCUDA.cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f($Ct *y, int *p, Array<$Ct,2> logx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            y[idx] = p[idx] > 0 ? -logx(p[idx]-1,idx) : 0;
        }
    }""")
    quote
        length(p) == size(logx,2) || throw("Length unmatch.")
        y = CuArray{T}(length(p))
        gdims, bdims = LibCUDA.cudims(length(y))
        culaunch($f, gdims, bdims, y.ptr, p.ptr, logx, Cint(length(y)))
        y
    end
end

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuVector{Cint}, gx::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = LibCUDA.cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f($Ct *gy, int *p, Array<$Ct,2> gx, Array<$Ct,2> logx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logx.length()) return;

        int subs[2];
        logx.ind2sub(subs, idx);
        int i = subs[0];
        int j = subs[1];
        if (p[j] > 0) {
            $Ct delta = (i == p[j]-1) ? 1 : 0;
            gx(i,j) += gy[j] * (exp(logx(i,j)) - delta);
        }
    }""")
    quote
        gdims, bdims = LibCUDA.cudims(length(logx))
        culaunch($f, gdims, bdims, gy.ptr, p.ptr, gx, logx)
    end
end
