export window

immutable Window{N}
    dims::Tuple{Vararg{Int,N}}
    strides::Tuple{Vararg{Int,N}}
    pads::Tuple{Vararg{Int,N}}
end

function Window{N}(dims::Tuple{Vararg{Int,N}}, stride, pad)
    strides = typeof(stride) == Int ? ntuple(_ -> stride, N) : stride
    pads = typeof(pad) == Int ? ntuple(_ -> pad, N) : pad
    Window(dims, strides, pads)
end

Base.size(w::Window) = w.dims
Base.size(w::Window, i::Int) = w.dims[i]
Base.size(x::AbstractArray, w::Window, i::Int) = (size(x,i) + 2*pad(w,i) - size(w,i)) ÷ stride(w,i) + 1
Base.strides(w::Window) = w.strides
Base.stride(w::Window, i::Int) = w.strides[i]
pads(w::Window) = w.pads
pad(w::Window, i::Int) = w.pads[i]

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32)
const WINDOW1D_F64 = Libdl.dlsym(libmerlin, :window1d_f64)
const WINDOW1D_I64 = Libdl.dlsym(libmerlin, :window1d_i64)
const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_f32)
const WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_f64)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32_grad)
const ∇WINDOW1D_F64 = Libdl.dlsym(libmerlin, :window1d_f64_grad)
const ∇WINDOW1D_I64 = Libdl.dlsym(libmerlin, :window1d_i64_grad)
const ∇WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_f32_grad)
const ∇WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_f64_grad)

chandle(w::Window{1}, ::Type{Float32}) = WINDOW1D_F32
chandle(w::Window{1}, ::Type{Float64}) = WINDOW1D_F64
chandle(w::Window{1}, ::Type{Int64}) = WINDOW1D_I64
chandle(w::Window{2}, ::Type{Float32}) = WINDOW2D_F32
chandle(w::Window{2}, ::Type{Float64}) = WINDOW2D_F64
chandle(w::Window{2}, ::Type{Int64}) = WINDOW2D_I64
∇chandle(w::Window{1}, ::Type{Float32}) = ∇WINDOW1D_F32
∇chandle(w::Window{1}, ::Type{Float64}) = ∇WINDOW1D_F64
∇chandle(w::Window{1}, ::Type{Int64}) = ∇WINDOW1D_I64
∇chandle(w::Window{2}, ::Type{Float32}) = ∇WINDOW2D_F32
∇chandle(w::Window{2}, ::Type{Float64}) = ∇WINDOW2D_F64
∇chandle(w::Window{2}, ::Type{Int64}) = ∇WINDOW2D_I64

"""
    window(x::Var, dims, [strides, pads])

* x::Var: input var
* dims::Tuple: window size
* strides::Tuple: stride size
* pads:Tuple: padding size

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = window(x, (10,))
```
"""
window(x, dims; stride=1, pad=0) = window(x, Window(dims,stride,pad))

@graph function window(x::Var, w::Window)
    y = window(x.data, w)
    df(gy) = isconst(x) || ∇window!(x.grad, gy, w)
    Var(y, [x], window, df)
end

function window{T}(x::Array{T}, w::Window{1})
    y = similar(x, size(w,1), size(vec(x),w,1))
    h = chandle(w, T)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        x, y, length(x), size(w,1), stride(w,1), pad(w,1))
    y
end

function ∇window!{T}(gx::Array{T}, gy::Array{T}, w::Window{1})
    h = ∇chandle(w, T)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        gx, gy, length(gx), size(w,1), stride(w,1), pad(w,1))
    gx
end
