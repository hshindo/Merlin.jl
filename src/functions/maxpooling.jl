export maxpooling

const MAXPOOLING2D_F32 = Libdl.dlsym(libmerlin, :maxpooling2d_float)
const MAXPOOLING2D_F64 = Libdl.dlsym(libmerlin, :maxpooling2d_double)
const ∇MAXPOOLING2D_F32 = Libdl.dlsym(libmerlin, :maxpooling2d_grad_float)
const ∇MAXPOOLING2D_F64 = Libdl.dlsym(libmerlin, :maxpooling2d_grad_double)

maxpooling_handle(::Type{Float32}) = MAXPOOLING2D_F32, ∇MAXPOOLING2D_F32
maxpooling_handle(::Type{Float64}) = MAXPOOLING2D_F64, ∇MAXPOOLING2D_F64

type Pooling{N}
    windims::NTuple{N,Int}
    stride::NTuple{N,Int}
    paddims::NTuple{N,Int}
end

function Pooling{N}(windims::NTuple{N,Int}, stride, paddims)
    length(stride) == 0 && (stride = ntuple(_ -> 1, N))
    length(paddims) == 0 && (paddims = ntuple(_ -> 0, N))
    Pooling(windims, stride, paddims)
end

"""
    maxpooling(window, [stride, padding])

## Arguments
* windims::NTuple{N,Int}: window size
* stride::NTuple{N,Int}: stride size. Default: (1,1,...)
* paddims::NTuple{N,Int}: padding size. Default: (0,0,...)

## 👉 Example
```julia
x = Var(rand(Float32,5,4,3,2))
y = maxpooling(x, (3,3), stride=(1,1), paddims=(0,0))
```
"""
function maxpooling{N}(x::Var, windims::NTuple{N,Int}; stride=(), paddims=())
    p = Pooling(windims, stride, paddims)
    if typeof(x.data) <: Array
        y, maxidxs = maxpooling(x.data, p)
        df(gy) = hasgrad(x) && ∇maxpooling(x.grad, gy, maxidxs, p)
    elseif typeof(x.data) <: CuArray
        throw("Not implemented yet.")
    end
    Var(y, [x], maxpooling, df)
end

function outsize{N}(x::UniArray, p::Pooling{N})
    Int[(size(x,i)+2*p.paddims[i]-p.windims[i]) ÷ p.stride[i] + 1 for i=1:N]
end

function maxpooling{T}(x::Array{T,3}, p::Pooling)
    h = maxpooling_handle(T)[1]
    outdims = outsize(x, p)
    y = Array(T, outdims...)
    maxidxs = Array(Cint, prod(outdims))
    ccall(h, Void, (Ptr{T},Ptr{Cint},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
        x, Cint[size(x)...], y, maxidxs, Cint[p.windims...], Cint[p.stride...], Cint[p.paddims...])
    y, maxidxs
end

maxpooling(x::CuArray, p::Pooling) = pooling(x, p.windims, p.paddims, p.stride,
    CUDNN_POOLING_MAX)

function ∇maxpooling{T}(gx::Array{T}, gy::Array{T}, maxidxs::Vector{Cint}, p::Pooling)
    h = maxpooling_handle(T)[2]
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint), gx, gy, maxidxs, length(gy))
    gx
end

function ∇maxpooling{T}(gx::CuArray{T}, gy::CuArray{T}, x::CuArray, y::CuArray,
    p::Pooling)

    ∇pooling!(y, gy, x, p.windims, p.paddims, p.stride, CUDNN_POOLING_MAX, gx)
end
