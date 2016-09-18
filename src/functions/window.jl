export window

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32)
const WINDOW1D_F64 = Libdl.dlsym(libmerlin, :window1d_f64)
const WINDOW1D_I64 = Libdl.dlsym(libmerlin, :window1d_i64)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32_grad)
const ∇WINDOW1D_F64 = Libdl.dlsym(libmerlin, :window1d_f64_grad)
const ∇WINDOW1D_I64 = Libdl.dlsym(libmerlin, :window1d_i64_grad)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
window1d_handle(::Type{Float64}) = WINDOW1D_F64
window1d_handle(::Type{Int64}) = WINDOW1D_I64
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32
∇window1d_handle(::Type{Float64}) = ∇WINDOW1D_F64
∇window1d_handle(::Type{Int64}) = ∇WINDOW1D_I64

#=
immutable Window{N}
    dims::Tuple{Vararg{Int,N}}
    strides::Tuple{Vararg{Int,N}}
    pads::Tuple{Vararg{Int,N}}
end

function Window{N}(dims::Tuple{Vararg{Int,N}}, strides, pads)
    typeof(strides) == Int && (strides = ntuple(_ -> strides, N))
    typeof(pads) == Int && (pads = ntuple(_ -> pads, N))
    Window(dims, strides, pads)
end

Base.size(w::Window) = w.dims
Base.size(w::Window, i::Int) = w.dims[i]
Base.strides(w::Window) = w.strides
Base.stride(w::Window, i::Int) = w.strides[i]
pads(w::Window) = w.pads
pad(w::Window, i::Int) = w.pads[i]
=#

function window{N}(x::AbstractArray, dims::Tuple{Vararg{Int,N}}; strides=nothing, pads=nothing)
    strides == nothing && (strides = Int[1 for i=1:N])
    pads == nothing && (pads = Int[0 for i=1:N])
    if N == 1
        window1d(x, dims[1], strides[1], pads[1])
    else
        throw("Not implemented yet")
    end
end

function window1d{T}(x::Array{T}, dims, strides, pads)
    y = similar(x, dims[1], prod(outsize(x,dims,strides,pads)))
    h = window1d_handle(T)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        x, y, length(x), dims[1], strides[1], pads[1])
    y
end

function outsize(x::AbstractArray, dims, strides, pads)
    Int[(size(x,i)+2*pads[i]-dims[i]) ÷ strides[i] + 1 for i=1:length(dims)]
end

function ∇window!{T}(gx::Array{T}, gy::Array{T}, w, s, p)
    ccall(∇chandle(w,T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        gx, gy, length(gx), w, s, p)
    gx
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
