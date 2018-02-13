export window1d

function window1d(x::Var, batchdims)
    y = window1d(x.data, batchdims)
    Var(y, (window1d,x,batchdims))
end
window1d(x::Node, batchdims) = Node(window1d, x, batchdims)

function window1d()
end

function window1d(x::Matrix{T}, batchdims::Vector{Int}, filtersize::Int, pad::Int, stride::Int, dilation::Int=1) where T
    n = size(x, 1)
    y = zeros(T, n*filtersize, sum(batchdims))
    yi = 1
    s = 1
    for dim in batchdims
        i = s - pad
        while i + filtersize <= s + dim + pad
            for w = 0:filtersize-1
                j = i + w * dilation
                if j >= s && j < s + dim
                    xi = (j-1) * n + 1
                    copy!(y, yi, x, xi, n)
                end
                yi += n
            end
            i += stride
        end
        s += dim
    end
    y
end

@generated function window1d(x::CuMatrix{T}, batchdims::Vector{Int}) where T
    Ct = cstring(T)
    f = CuFunction("""
    __global__ void window1d($Ct *y, int sizeY, $Ct *w, int *x, int n) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= sizeY) return;

        int j = idxY / n;
        int i = idxY - n * j;
        y[idxY] = w[(x[j]-1) * n + i];
    }""")
    quote

    end
end
