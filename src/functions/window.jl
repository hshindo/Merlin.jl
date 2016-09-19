export window

immutable Window{N}
    dims::Tuple{Vararg{Int,N}}
    strides::Tuple{Vararg{Int,N}}
    pads::Tuple{Vararg{Int,N}}
end

function Window{N}(dims::Tuple{Vararg{Int,N}}, strides=nothing, pads=nothing)
    strides == nothing && (strides = ntuple(_ -> 1, N))
    pads == nothing && (pads = ntuple(_ -> 0, N))
    Window(dims, strides, pads)
end

Base.size(w::Window) = w.dims
Base.size(w::Window, i::Int) = w.dims[i]
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

function window(x::Var, dims; strides=nothing, pads=nothing)
    w = Window(dims, strides, pads)
    y = window(w, x.data)
    df(gy) = isconstant(x) || ∇window!(w, x.grad, gy)
    Var(y, [x], w, df)
end

function window(x::AbstractArray, dims; strides=nothing, pads=nothing)
    w = Window(dims, strides, pads)
    window(w, x)
end

function window{T}(w::Window{1}, x::Array{T})
    y = similar(x, size(w,1), prod(outsize(w,x)))
    h = chandle(w, T)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        x, y, length(x), size(w,1), stride(w,1), pad(w,1))
    y
end

function ∇window!{T}(w::Window{1}, gx::Array{T}, gy::Array{T})
    h = ∇chandle(w, T)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        gx, gy, length(gx), w, s, p)
    gx
end

function outsize{N}(w::Window{N}, x::AbstractArray)
    Int[(size(x,i)+2*pad(w,i)-size(w,i)) ÷ stride(w,i) + 1 for i=1:N]
end

#=
const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_float)
const WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_double)
const ∇WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_grad_float)
const ∇WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_grad_double)

"""
    window(x::Var, dims, [stride, padding])

* x::Var: input var
* dims::Tuple: window size
* stride::Union{Int,Tuple}: stride size
* padding:Union{Int,Tuple}: padding size

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = window(x, (10,2), stride=1, padding=0)
```
"""
window(x::Var, dims; stride=1, padding=0) = Window(dims,stride,padding)(x)

type Window{N}
    dims::Tuple{Vararg{Int,N}}
    stride::Tuple{Vararg{Int,N}}
    padding::Tuple{Vararg{Int,N}}
end

handle(::Window{2}, ::Type{Float32}) = WINDOW2D_F32
handle(::Window{2}, ::Type{Float64}) = WINDOW2D_F64
∇handle(::Window{2}, ::Type{Float32}) = ∇WINDOW2D_F32
∇handle(::Window{2}, ::Type{Float64}) = ∇WINDOW2D_F64

function Window(dims::Tuple, stride=1, padding=0)
    N = length(dims)
    typeof(stride) == Int && (stride = ntuple(_ -> stride, N))
    typeof(padding) == Int && (padding = ntuple(_ -> padding, N))
    Window(dims, stride, padding)
end

function (w::Window{N}){N}(x::Var)
    y = w(x.data)
    df(gy) = hasgrad(x) && ∇window!(w, x.grad, gy)
    Var(y, [x], w, df)
end

function (w::Window{N}){T,N}(x::Array{T})
    h = handle(w, T)
    outdims = outsize(w, x)
    y = Array(T, prod(w.dims)*size(x,N+1), prod(outdims))
    xsize = Cint[size(x)...]
    ndims(x) == N && push!(xsize, Cint(1))
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    x, y, xsize, Cint[w.dims...], Cint[w.stride...], Cint[w.padding...])
    y
end

function outsize{N}(w::Window{N}, x::UniArray)
    dims = Array(Int, N)
    for i = 1:N
        dims[i] = (size(x,i) + 2*w.padding[i] - w.dims[i]) ÷ w.stride[i] + 1
    end
    dims
end

function ∇window!{T,N}(w::Window{N}, gx::Array{T}, gy::Array{T})
    h = handle(w, T)
    xsize = Cint[size(gx)...]
    ndims(gx) == N && push!(xsize, Cint(1))
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    gx, gy, xsize, Cint[w.dims...], Cint[w.stride...], Cint[w.padding...])
end
=#
