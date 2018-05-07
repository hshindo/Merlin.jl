export Conv1d

doc"""
    Conv1d(T, ksize, inchannel, outchannel, kwargs...)

1-dimensional convolution function.

```julia
T = Float32
x = Var(rand(T,10,5))
f = Conv1d(T, 5, 10, 3, pad=2)
y = f(x)
```
"""
mutable struct Conv1d <: Functor
    W::Var
    b::Var
    ksize::Int
    pad::Int
    stride::Int
    dilation::Int
end

getparams(f::Conv1d) = (f.W, f.b)

function Conv1d(::Type{T}, ksize::Int, inchannel::Int, outchannel::Int;
    pad=0, stride=1, dilation=1, init_W=Xavier(), init_b=Fill(0)) where T

    W = init_W(T, ksize*inchannel, outchannel)
    b = init_b(T, outchannel)
    Conv1d(param(W), param(b), ksize, pad, stride, dilation)
end

function (f::Conv1d)(x::Var, batchdims::Vector{Int})
    @assert ndims(x) == 2 && sum(batchdims) == size(x,ndims(x))
    idx = conv1d_index(batchdims, f.ksize, f.pad, f.stride, f.dilation)
    h = lookup(x, idx)
    linear(h, f.W, f.b)
end

function conv1d_index(batchdims::Vector{Int}, ksize::Int, pad::Int, stride::Int, dilation::Int)
    outdims = map(batchdims) do d
        k = (ksize - 1) * dilation + 1
        (d + 2pad - k) ÷ stride + 1
    end
    cumdim = 0
    y = Array{Int}(ksize, sum(outdims))
    yi = 1
    for n = 1:length(batchdims)
        ndims = batchdims[n]
        i = cumdim - pad + 1
        for d = 1:outdims[n]
            for j = i:dilation:i+(ksize-1)*dilation
                xi = cumdim < j <= cumdim+ndims ? j : 0
                y[yi] = xi
                yi += 1
            end
            i += stride
        end
        cumdim += ndims
    end
    y
end
