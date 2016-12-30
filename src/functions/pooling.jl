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
function pooling{N}(mode, x::Var, dims::NTuple{N,Int}; pads=nothing, strides=nothing)
    pads == nothing && (pads = ntuple(_ -> 0, N))
    strides == nothing && (strides = ntuple(_ -> 1, N))
    if mode == :max
        f = maxpooling
    elseif mode == :average
        f = avgpooling
    else
        throw("Invalid mode is specified: $(mode).")
    end
    f(x, dims, pads, strides)
end

function maxpooling{T}(x::Var{Array{T,4}}, dims, pads, strides)
    h = maxpooling2d_handle(T)
    spdims = ntuple(i -> (size(x.data,i) + 2pads[i] - dims[i]) ÷ strides[i] + 1, 2)
    y = Array(T, spdims..., size(x,3), size(x,4))
    inds = Array(Cint, length(y))
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        x.data, y, inds, size(x.data,1), size(x.data,2), size(x.data,3)*size(x.data,4),
        dims[1], dims[2], pads[1], pads[2], strides[1], strides[2])

    function df{T}(gy::Array{T})
        isvoid(x.grad) && return
        ccall(∇maxpooling2d_handle(T), Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint),
            gy, x.grad, inds, length(gy))
    end
    Var(y, df, (x,))
end

function avgpooling{T}(x::Var{Array{T,4}}, dims, pads, strides)
    h = avgpooling2d_handle(T)
    spdims = ntuple(i -> (size(x.data,i) + 2pads[i] - dims[i]) ÷ strides[i] + 1, 2)
    y = Array(T, spdims..., size(x,3), size(x,4))
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        x.data, y, size(x.data,1), size(x.data,2), size(x.data,3)*size(x.data,4),
        dims[1], dims[2], pads[1], pads[2], strides[1], strides[2])

    function df{T}(gy::Array{T,4})
        isvoid(x.grad) && return
        gx = x.grad
        h = ∇avgpooling2d_handle(T)
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            gy, gx, size(gx,1), size(gx,2), size(gx,3)*size(gx,4),
            dims[1], dims[2], pads[1], pads[2], strides[1], strides[2])
    end
    Var(y, df, (x,))
end
