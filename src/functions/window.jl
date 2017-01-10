export window

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32_grad)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

"""
    window(x::Var, dims, [pads, strides])

* x::Var: input var
* dims::Tuple: window size
* pads:Tuple: padding size
* strides::Tuple: stride size

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = window(x, (10,), pads=(0,), strides=(1,))
```
"""
function window{N}(x::Var, dims::NTuple{N,Int}, pads::NTuple{N,Int}, strides::NTuple{N,Int})
    isa(x.data, Void) && return Var(nothing, window, (x,dims,pads,strides))

    y = window(x.data, dims, pads, strides)
    df(gy) = isa(x.grad, Void) || ∇window!(gy, x.grad, dims, pads, strides)
    Var(y, df, (x,))
end

function window{N}(x, dims::NTuple{N,Int}; pads=nothing, strides=nothing)
    pads == nothing && (pads = ntuple(_ -> 0, N))
    strides == nothing && (strides = ntuple(_ -> 1, N))
    window(x, dims, pads, strides)
end

function window{T}(x::Array{T}, dims::NTuple{1,Int}, pads, strides)
    c = (length(x) + 2pads[1] - dims[1]) ÷ strides[1] + 1
    y = Array{T}(dims[1], c)
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        x, y, length(x), dims[1], pads[1], strides[1])
    y
end

function ∇window!{T}(gy::Array{T}, gx::Array{T}, dims::NTuple{1,Int}, pads, strides)
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        gy, gx, length(gx), dims[1], pads[1], strides[1])
end
