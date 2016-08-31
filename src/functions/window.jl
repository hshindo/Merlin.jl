export window

const WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_float)
const WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_double)
const ∇WINDOW2D_F32 = Libdl.dlsym(libmerlin, :window2d_grad_float)
const ∇WINDOW2D_F64 = Libdl.dlsym(libmerlin, :window2d_grad_double)

"""
    window(x::Var, dims::Tuple, [stride, padding])

* dims: window size
* stride::Union{Int,Tuple}: stride size.
* padding:Union{Int,Tuple}: padding size.

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

handle(::Type{Window{2}}, ::Type{Float32}) = WINDOW2D_F32
handle(::Type{Window{2}}, ::Type{Float64}) = WINDOW2D_F64
∇handle(::Type{Window{2}}, ::Type{Float32}) = ∇WINDOW2D_F32
∇handle(::Type{Window{2}}, ::Type{Float64}) = ∇WINDOW2D_F64

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
    h = handle(Window{N}, T)
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
    h = handle(Window{N}, T)
    xsize = Cint[size(gx)...]
    ndims(gx) == N && push!(xsize, Cint(1))
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    gx, gy, xsize, Cint[w.dims...], Cint[w.stride...], Cint[w.padding...])
end

#=
function window{T,N}(p::Window{N}, x::Array{T})
    #@assert ndims(x) == N+1
    h = handle(Window{N}, T)
    outdims = outsize(x, winsize, stride, padsize)
    y = Array(T, prod(outdims), prod(winsize)*size(x,N+1))
    xsize = Cint[size(x)...]
    ndims(x) == N && push!(xsize, Cint(1))
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    x, y, xsize, Cint[winsize...], Cint[stride...], Cint[padsize...])
    y
end

function outsize(x, winsize, stride, padsize)
    N = length(winsize)
    dims = Array(Int, N)
    for i = 1:N
        dims[i] = (size(x,i) + 2*padsize[i] - winsize[i]) ÷ stride[i] + 1
    end
    dims
end

function ∇window!{T,N}(winsize::NTuple{N,Int}, stride, padsize, gx::Array{T}, gy::Array{T})
    h = handle(Window{N}, T)
    xsize = Cint[size(gx)...]
    ndims(gx) == N && push!(xsize, Cint(1))
    ccall(h, Void, (Ptr{T},Ptr{T},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    gx, gy, xsize, Cint[winsize...], Cint[stride...], Cint[padsize...])
end
=#
