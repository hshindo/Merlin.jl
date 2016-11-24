export window

immutable Window{N}
    dims::NTuple{N,Int}
    pads::NTuple{N,Int}
    strides::NTuple{N,Int}
end

Base.size(w::Window) = w.dims
Base.size(w::Window, d::Int) = w.dims[d]
Base.size(x::AbstractArray, w::Window, i::Int) = (size(x,i) + 2*pad(w,i) - size(w,i)) ÷ stride(w,i) + 1
Base.strides(w::Window) = w.strides
Base.stride(w::Window, i::Int) = w.strides[i]

pads(w::Window) = w.pads
pad(w::Window, i::Int) = w.pads[i]

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32)
const WINDOW1D_I64 = Libdl.dlsym(libmerlin, :window1d_i64)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32_grad)
const ∇WINDOW1D_I64 = Libdl.dlsym(libmerlin, :window1d_i64_grad)

chandle(w::Window{1}, ::Type{Float32}) = WINDOW1D_F32
chandle(w::Window{1}, ::Type{Int64}) = WINDOW1D_I64
∇chandle(w::Window{1}, ::Type{Float32}) = ∇WINDOW1D_F32
∇chandle(w::Window{1}, ::Type{Int64}) = ∇WINDOW1D_I64

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
window(x, dims::Tuple{Int}; pads=(0,), strides=(1,)) = window(x, Window(dims,pads,strides))
window(x, dims::Tuple{Int,Int}; pads=(0,0), strides=(1,1)) = window(x, Window(dims,pads,strides))
window(x, dims::Tuple{Int,Int,Int}; pads=(0,0,0), strides=(1,1,1)) = window(x, Window(dims,pads,strides))

function window(x::Var, w::Window)
    x.data == nothing && return Var(nothing, window, (x,w))
    y = window(x.data, w)
    df(gy) = isconst(x) || ∇window!(gy, x.grad, w)
    Var(y, window, (x,), df)
end

function window{T}(x::Array{T}, w::Window{1})
    y = Array{T}(size(w,1), size(vec(x),w,1))
    ccall(chandle(w,T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        x, y, length(x), size(w,1), pad(w,1), stride(w,1))
    y
end

function ∇window!{T}(gy::Array{T}, gx::Array{T}, w::Window{1})
    ccall(∇chandle(w,T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        gx, gy, length(gx), size(w,1), pad(w,1), stride(w,1))
end
