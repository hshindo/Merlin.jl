@generated function softmax_crossentropy(p::CuVector{Cint}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void softmax_crossentropy($Ct *y, int *p, Array<$Ct,2> logx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            y[idx] = p[idx] > 0 ? -logx(p[idx]-1,idx) : 0;
        }
    }""")
    quote
        length(p) == size(logx,2) || throw("Length unmatch.")
        y = CuArray{T}(length(p))
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, Ptr{T}(y), Ptr{Cint}(p), logx, Cint(length(y)))
        y
    end
end

@generated function softmax_crossentropy(p::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f($Ct *y, $Ct *p, $Ct *logx, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            y[idx] = -p[idx] * logx[idx];
        }
    }""")
    quote
        size(p) == size(logx) || throw("Length unmatch.")
        y = similar(p)
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, Ptr{T}(y), Ptr{T}(p), Ptr{T}(logx), length(y))
        vec(sum(y,1))
    end
end

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuVector{Cint}, gx::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f($Ct *gy, int *p, Array<$Ct,2> gx, Array<$Ct,2> logx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logx.length()) return;

        int ndIdx[2];
        logx.idx2ndIdx(ndIdx, idx);
        int i = ndIdx[0];
        int j = ndIdx[1];
        if (p[j] > 0) {
            $Ct delta = (i == p[j]-1) ? 1 : 0;
            gx(i,j) += gy[j] * (exp(logx(i,j)) - delta);
        }
    }""")
    quote
        gdims, bdims = cudims(length(logx))
        culaunch($f, gdims, bdims, Ptr{T}(gy), Ptr{Cint}(p), gx, logx)
    end
end

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuMatrix{T}, gx::CuMatrix{T}, logx::CuMatrix{T}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f($Ct *gy, $Ct *p, $Ct *gx, Array<$Ct,2> logx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logx.length()) return;

        int ndIdx[2];
        logx.idx2ndIdx(ndIdx, idx);
        int i = ndIdx[0];
        int j = ndIdx[1];
        gx[idx] += gy[j] * (exp(logx[idx]) - p[idx]);
    }""")
    quote
        gdims, bdims = cudims(length(logx))
        culaunch($f, gdims, bdims, Ptr{T}(gy), Ptr{Cint}(p), Ptr{T}(gx), logx)
    end
end
