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

function (f::Conv1d)(v::Var)
    x = cat(2, v)
    batchsize = map(x -> size(x,2), xs)
    idx = conv1d_index(batchsize, f.ksize, f.pad, f.stride, f.dilation)
    h = lookup(x, idx)
    y = linear(h, f.W, f.b)
    ysize = map(xs) do x
        k = (f.ksize - 1) * f.dilation + 1
        (size(x,2) + 2*f.pad - k) ÷ f.stride + 1
    end
    unsafe_split(y, ysize)
end
(f::Conv1d)(x::Node) = Node(f, x)

function conv1d_index(ksize::Int, pad::Int, stride::Int, dilation::Int, batchsize::Vector{Int})
    outdims = map(batchsize) do d
        k = (ksize - 1) * dilation + 1
        (d + 2pad - k) ÷ stride + 1
    end
    cumdim = 0
    y = Array{Int}(ksize, sum(outdims))
    yi = 1
    for n = 1:length(batchsize)
        ndims = batchsize[n]
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
