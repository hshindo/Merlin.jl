export Conv
import Base.conv

const IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32)
const ∇IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32_grad)

im2col_handle(::Type{Float32}) = IM2COL_F32
∇im2col_handle(::Type{Float32}) = ∇IM2COL_F32

"""
    Conv(T::Type, dims::Int..., [padding], [strides])

N-dimensional convolution function.

* dims: ``k1, k2, ..., c_{in}, c_{out}`` where ``k1``, ``k2``, ... are kernel size.
``c_{in}`` and ``c_{out}`` are input channel and output channel, respectively.
* [padding=(0,0,...)]: spatial padding size.
* [strides=(1,1,...)]: kernel strides.

```julia
T = Float32
x = Var(rand(T,5,4,3,2))
f = Conv(T,2,2,3,4, padding=(0,0), strides=(1,1))
y = f(x)
```
"""
type Conv{N}
    w::Var
    b::Var
    padding::NTuple{N,Int}
    strides::NTuple{N,Int}
end

function Conv{T}(::Type{T}, dims::Int...; padding=nothing, strides=nothing)
    w = uniform(T, -0.01, 0.01, dims)
    b = zeros(T, dims[end])
    N = length(dims) - 2
    padding == nothing && (padding = ntuple(_ -> 0, N))
    strides == nothing && (strides = ntuple(_ -> 1, N))
    Conv(zerograd(w), zerograd(b), padding, strides)
end

(f::Conv)(x::Var{Void}) = Var(Void(), f, (x,))

function (f::Conv){T<:Array}(x::Var{T})
    w, b, padding, strides = f.w, f.b, f.padding, f.strides
    y, work = conv(x.data, w.data, b.data, padding, strides)
    function df(gy::Array)
        gy = permutedims(gy, [1,2,4,3])
        gy_mat = reshape(gy, size(gy,1)*size(gy,2)*size(gy,3), size(gy,4))
        isvoid(x.grad) || ∇conv_x!(gy_mat, x.grad, w.data, work, padding, strides)
        isvoid(w.grad) || ∇conv_w!(gy_mat, w.grad, work)
        isvoid(b.grad) || BLAS.axpy!(eltype(gy)(1), sum(gy_mat,1), b.grad)
    end
    Var(y, df, (x,w,b))
end

function conv{T}(x::Array{T,4}, w::Array{T,4}, b::Vector{T}, padding::NTuple{2,Int}, strides::NTuple{2,Int})
    size(x,3) == size(w,3) || throw("Input channel size mismatch.")

    outsize = ntuple(i -> (size(x,i)+2padding[i]-size(w,i)) ÷ strides[i] + 1, 2)
    work = ones(T, prod(outsize)*size(x,4), size(w,1)*size(w,2)*size(w,3))
    ccall(im2col_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        x, work, size(x,1), size(x,2), size(x,3), size(x,4),
        size(w,1), size(w,2), padding[1], padding[2], strides[1], strides[2])

    _w = reshape(w,size(work,2),size(w,4))
    y = work * _w
    y = reshape(y, outsize..., size(x,4), size(w,4))
    y = permutedims(y, [1,2,4,3])
    broadcast!(.+, y, y, b)
    y, work
end

function ∇conv_x!{T}(gy_mat::Matrix{T}, gx::Array{T,4}, w::Array{T,4}, work::Matrix{T}, padding, strides)
    w_mat = reshape(w, size(work,2), size(w,4))
    gwork = similar(work)
    BLAS.gemm!('N', 'T', T(1), gy_mat, w_mat, T(0), gwork)
    ccall(∇im2col_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        gwork, gx, size(gx,1), size(gx,2), size(gx,3), size(gx,4),
        size(w,1), size(w,2), padding[1], padding[2], strides[1], strides[2])
end

function ∇conv_w!{T}(gy_mat::Matrix{T}, gw::Array{T,4}, work::Matrix{T})
    gw_mat = reshape(gw, size(work,2), size(gw,4))
    BLAS.gemm!('T', 'N', T(1), work, gy_mat, T(1), gw_mat)
end
