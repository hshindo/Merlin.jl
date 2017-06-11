export pooling

const MAXPOOLING2D_F32 = Libdl.dlsym(libmerlin, :maxpooling2d_f32)
const AVGPOOLING2D_F32 = Libdl.dlsym(libmerlin, :avgpooling2d_f32)
const ∇MAXPOOLING2D_F32 = Libdl.dlsym(libmerlin, :maxpooling2d_f32_grad)
const ∇AVGPOOLING2D_F32 = Libdl.dlsym(libmerlin, :avgpooling2d_f32_grad)

maxpooling2d_handle(::Type{Float32}) = MAXPOOLING2D_F32
avgpooling2d_handle(::Type{Float32}) = AVGPOOLING2D_F32
∇maxpooling2d_handle(::Type{Float32}) = ∇MAXPOOLING2D_F32
∇avgpooling2d_handle(::Type{Float32}) = ∇AVGPOOLING2D_F32

"""
    pooling(mode, x::Var, dims::Tuple, [pads], [stride])

N-dimensional pooling function.

* mode: `:max` or `:average`
* dims: kernel size.
* [pads=(0,0,...)]: spatial padding size.
* [strides=(1,1,...)]: kernel strides.

```julia
x = Var(rand(Float32,5,4,3,2))
y = pooling(:max, x, (2,2), pads=(0,0), strides=(1,1))
```
"""
function pooling{N}(mode::Symbol, x::Var, dims::NTuple{N,Int}; pads=nothing, strides=nothing)
    pads == nothing && (pads = ntuple(_ -> 0, N))
    strides == nothing && (strides = ntuple(_ -> 1, N))
    if mode == :max
        f = maxpooling
    elseif mode == :average
        f = avgpooling
    else
        throw("Invalid mode is specified: $(mode).")
    end
    isvoid(x.data) && return Var(nothing, f, (x,dims,pads,strides))
    iscuda(x.data) && return CUDA.f(x, dims, pads, strides)
    f(x, dims, pads, strides)
end

function maxpooling(x::Var, dims::NTuple{2,Int}, pads, strides)
    T = eltype(x.data)
    h = maxpooling2d_handle(T)
    spdims = ntuple(i -> (size(x.data,i) + 2pads[i] - dims[i]) ÷ strides[i] + 1, 2)
    y = Array(T, spdims..., size(x,3), size(x,4))
    inds = Array(Cint, length(y))
    function f{T}(x::Array{T})
        ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            x, y, inds, size(x,1), size(x,2), size(x,3)*size(x,4),
            dims[1], dims[2], pads[1], pads[2], strides[1], strides[2])
    end
    f(x.data)

    function df{T}(gy::Array{T})
        isvoid(x.grad) && return
        ccall(∇maxpooling2d_handle(T), Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint),
            gy, x.grad, inds, length(gy))
    end
    Var(y, df, (x,))
end

function avgpooling(x::Var, dims::NTuple{2,Int}, pads, strides)
    T = eltype(x.data)
    h = avgpooling2d_handle(T)
    spdims = ntuple(i -> (size(x.data,i) + 2pads[i] - dims[i]) ÷ strides[i] + 1, 2)
    y = Array(T, spdims..., size(x,3), size(x,4))
    function f{T}(x::Array{T})
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            x, y, size(x,1), size(x,2), size(x,3)*size(x,4),
            dims[1], dims[2], pads[1], pads[2], strides[1], strides[2])
    end
    f(x.data)

    function df{T}(gy::Array{T})
        isvoid(x.grad) && return
        gx = x.grad
        h = ∇avgpooling2d_handle(T)
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            gy, gx, size(gx,1), size(gx,2), size(gx,3)*size(gx,4),
            dims[1], dims[2], pads[1], pads[2], strides[1], strides[2])
    end
    Var(y, df, (x,))
end
