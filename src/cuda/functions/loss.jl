@generated function softmax_crossentropy!(out, p::CuVector{Cint}, x::CuMatrix{T}) where T
    Ct = LibCUDA.cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)

    __global__ void f(Array<$Ct,1> y, int *p, Array<$Ct,2> logx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < y.length()) {
            y(idx) = p[idx] > 0 ? -logx(p[idx]-1,idx) : 0;
        }
    }""")
    quote
        length(p) == size(x,2) || throw("Length unmatch.")
        logx = logsoftmax(x)
        y = CuArray{T}(length(p))
        gdims, bdims = LibCUDA.cudims(length(y))
        culaunch($f, gdims, bdims, y, p.ptr, logx)

        out.data = y
        out.∇! = () -> begin
            isvoid(out[1].grad) && return
            ∇softmax_crossentropy!(out.grad, p, logx, out[1].grad)
        end
        out
    end
end

@generated function ∇softmax_crossentropy!(gy::CuMatrix{T}, p::CuVector{Cint}, logq::CuMatrix{T}, gq::CuMatrix{T}) where T
    f = CuFunction("""
    __global__ void f($Ct *gy, int *p, Array<$Ct,2> logq, Array<$Ct,2> gq) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= logq.length()) return;

        int subs[2];
        logq.ind2sub(subs);
        int i = subs[0];
        int j = subs[1];
        if (p[j] > 0) {
            $Ct delta = (i == p[j]-1) ? 1 : 0;
            gq(i,j) += gy[j] * (exp(logq(i,j)) - delta);
        }
    }""")
    quote
        $f(gy.ptr, p.ptr, logq, gq, dx=length(logq))
    end
end

function ∇softmax_crossentropy2!(gy::Vector{T}, p::Vector{Int}, logq::Matrix{T}, gq::Matrix{T}) where T
    @inbounds for j = 1:length(p)
        p[j] > 0 || continue
        for i = 1:size(logq,1)
            delta = i == p[j] ? T(1) : T(0)
            gq[i,j] += gy[j] * (exp(logq[i,j]) - delta)
        end
    end
end
