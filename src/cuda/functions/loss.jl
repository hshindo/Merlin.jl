@generated function softmax_crossentropy!(out, p::CuVector{Cint}, x::CuMatrix{T}) where T
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
        length(p) == size(x,2) || throw("Length unmatch.")
        logx = logsoftmax(x)
        y = CuArray{T}(length(p))
        gdims, bdims = LibCUDA.cudims(length(y))
        culaunch($f, gdims, bdims, y.ptr, p.ptr, logx, Cint(length(y)))

        out.data = y
        out.work = (logx,)
        out
    end
end

function ∇softmax_crossentropy3!(gy::Vector{T}, p::Vector{Int}, gq::Matrix{T}, logq::Matrix{T}) where T
    @inbounds for j = 1:length(p)
        p[j] > 0 || continue
        for i = 1:size(logq,1)
            delta = i == p[j] ? T(1) : T(0)
            gq[i,j] += gy[j] * (exp(logq[i,j]) - delta)
        end
    end
end

@generated function ∇softmax_crossentropy!(gy::CuVector{T}, p::CuVector{Cint}, gq::CuMatrix{T}, logq::CuMatrix{T}) where T
    Ct = LibCUDA.cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f($Ct *gy, int *p, Array<$Ct,2> gq, Array<$Ct,2> logq) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logq.length()) return;

        int subs[2];
        logq.ind2sub(idx, subs);
        int i = subs[0];
        int j = subs[1];
        if (p[j] > 0) {
            $Ct delta = (i == p[j]-1) ? 1 : 0;
            gq(i,j) += gy[j] * (exp(logq(i,j)) - delta);
        }
    }""")
    quote
        gdims, bdims = LibCUDA.cudims(length(logq))
        culaunch($f, gdims, bdims, gy.ptr, p.ptr, gq, logq)
    end
end
