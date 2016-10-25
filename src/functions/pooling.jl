export maxpooling, avgpooling

const MAXPOOLING2D_F32 = Libdl.dlsym(libmerlin, :maxpooling2d_f32)
#const AVGPOOLING2D_F32 = Libdl.dlsym(libmerlin, :meanpooling2d_f32)
const ∇MAXPOOLING2D_F32 = Libdl.dlsym(libmerlin, :maxpooling2d_f32_grad)
#const ∇AVGPOOLING2D_F32 = Libdl.dlsym(libmerlin, :meanpooling2d_f32_grad)

maxpooling_handle(::Type{Float32}) = MAXPOOLING2D_F32
avgpooling_handle(::Type{Float32}) = AVGPOOLING2D_F32
∇maxpooling_handle(::Type{Float32}) = ∇MAXPOOLING2D_F32
∇avgpooling_handle(::Type{Float32}) = ∇AVGPOOLING2D_F32

"""
    maxpooling(windims, padding, stride)

```julia
x = Var(rand(Float32,5,4,3,2))
y = maxpooling(x, (3,3), (0,0), (1,1))
```
"""
function maxpooling(x::Var, windims, padding, stride)
    p = Pooling(windims, padding, stride)
    if typeof(x.data) <: Array
        y, maxidxs = maxpooling(x.data, p)
        df(gy) = isconst(x) || ∇maxpooling(x.grad, gy, maxidxs, p)
    else
        y = CUDNN.pooling(CUDNN_POOLING_MAX, windims, padding, stride, x.data)
        df(gy) = isconst(x) || CUDNN.∇pooling!(CUDNN_POOLING_MAX, windims, padding, stride, y.data, y.grad, x.data, x.grad)
    end
    Var(y, [x], maxpooling, df)
end

#=
export maxpooling, avgpooling

const MAXPOOLING2D_F32 = Libdl.dlsym(libmerlin, :maxpooling2d_f32)
#const AVGPOOLING2D_F32 = Libdl.dlsym(libmerlin, :meanpooling2d_f32)
const ∇MAXPOOLING2D_F32 = Libdl.dlsym(libmerlin, :maxpooling2d_f32_grad)
#const ∇AVGPOOLING2D_F32 = Libdl.dlsym(libmerlin, :meanpooling2d_f32_grad)

maxpooling_handle(::Type{Float32}) = MAXPOOLING2D_F32
avgpooling_handle(::Type{Float32}) = AVGPOOLING2D_F32
∇maxpooling_handle(::Type{Float32}) = ∇MAXPOOLING2D_F32
∇avgpooling_handle(::Type{Float32}) = ∇AVGPOOLING2D_F32

type Pooling{N}
    windims::NTuple{N,Int}
    padding::NTuple{N,Int}
    stride::NTuple{N,Int}
end

"""
    maxpooling(windims, padding, stride)

```julia
x = Var(rand(Float32,5,4,3,2))
y = maxpooling(x, (3,3), (0,0), (1,1))
```
"""
function maxpooling(x::Var, windims, padding, stride)
    p = Pooling(windims, padding, stride)
    if typeof(x.data) <: Array
        y, maxidxs = maxpooling(x.data, p)
        df(gy) = isconst(x) || ∇maxpooling(x.grad, gy, maxidxs, p)
    else
        y = CUDNN.pooling(CUDNN_POOLING_MAX, windims, padding, stride, x.data)
        df(gy) = isconst(x) || CUDNN.∇pooling!(CUDNN_POOLING_MAX, windims, padding, stride, y.data, y.grad, x.data, x.grad)
    end
    Var(y, [x], maxpooling, df)
end

function maxpooling{T}(x::Array{T,4}, windims, padding, stride)
    h = maxpooling_handle(T)
    outdims = Int[(size(x,i) + 2padding[i] - windims[i]) ÷ stride[i] + 1 for i=1:2]
    y = Array(T, outdims...)
    maxidxs = Array(Cint, prod(outdims))
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
        x, y, maxidxs, size(x,1), size(x,2), size(x,3), windims[1], windims[2], padding[1], padding[2], stride[1], stride[2])
    y, maxidxs
end

function ∇maxpooling{T}(gx::Array{T}, gy::Array{T}, maxidxs::Vector{Cint}, p::Pooling)
    h = maxpooling_handle(T)[2]
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint), gx, gy, maxidxs, length(gy))
    gx
end
=#
