export window

immutable Window{N}
    dims::Tuple{Vararg{Int,N}}
    strides::Tuple{Vararg{Int,N}}
    pads::Tuple{Vararg{Int,N}}
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
    window(x::Var, dims, [strides, pads])

* x::Var: input var
* dims::Tuple: window size
* strides::Tuple: stride size
* pads:Tuple: padding size

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = window(x, (10,), strides=(1,), pads=(0,))
```
"""
window(x, dims::Tuple{Int}; strides=(1,), pads=(0,)) = window(x, Window(dims,strides,pads))
window(x, dims::Tuple{Int,Int}; strides=(1,1), pads=(0,0)) = window(x, Window(dims,strides,pads))
window(x, dims::Tuple{Int,Int,Int}; strides=(1,1,1), pads=(0,0,0)) = window(x, Window(dims,strides,pads))

function window(x::Var, w::Window)
    x.data == nothing && return Var(nothing, (window,x,w))
    y = Var(eltype(x), (size(w,1),size(vec(x.data),w,1)), (x,))
    window!(x.data, w, y.data)
    y.df = () -> isconst(x) || ∇window!(y.grad, x.grad, w)
    y
end

function window!{T}(x::Array{T}, w::Window{1}, y::Array{T})
    ccall(chandle(w,T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        x, y, length(x), size(w,1), stride(w,1), pad(w,1))
end

function ∇window!{T}(gy::Array{T}, gx::Array{T}, w::Window{1})
    ccall(∇chandle(w,T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        gx, gy, length(gx), size(w,1), stride(w,1), pad(w,1))
end
