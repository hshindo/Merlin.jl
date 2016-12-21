export Conv
import Base.conv

const IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32)
const ∇IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32_grad)

im2col_handle(::Type{Float32}) = IM2COL_F32
∇im2col_handle(::Type{Float32}) = ∇IM2COL_F32

"""
    Conv(T::Type, filtersize::Tuple, [padding=0], [strides=1])

N-dimensional convolution function.

```julia
T = Float32
x = Var(rand(T,5,4,3,2))
f = Conv(T, (2,2,3,4), padding=(0,0), strides=(1,1))
y = f(x)
```
"""
type Conv{N}
    w::Var
    padding::NTuple{N,Int}
    strides::NTuple{N,Int}
end

function Conv{T,N}(::Type{T}, filtersize::NTuple{N,Int}; padding=0, strides=1)
    w = rand(T, filtersize)
    w .*= 0.02
    w .-= 0.01
    typeof(padding) == Int && (padding = ntuple(_ -> padding, N-2))
    typeof(strides) == Int && (strides = ntuple(_ -> strides, N-2))
    Conv(zerograd(w), padding, strides)
end

(f::Conv)(x::Var{Void}) = Var(Void(), f, (x,))

function (f::Conv){T<:Array}(x::Var{T})
    w, padding, strides = f.w, f.padding, f.strides
    y, work = conv(x.data, w.data, padding, strides)
    df(gy::Array) = isvoid(x.grad) || ∇conv!(gy, x.grad, w.data, w.grad, work, padding, strides)
    Var(y, df, (w,x))
end

function conv{T}(x::Array{T,4}, w::Array{T,4}, padding::NTuple{2,Int}, strides::NTuple{2,Int})
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
    y, work
end

function ∇conv!{T}(gy::Array{T,4}, gx::Array{T,4}, w::Array{T,4}, gw::Array{T,4}, work, padding, strides)
    outsize = ntuple(i -> (size(gx,i)+2padding[i]-size(w,i)) ÷ strides[i] + 1, 2)
    gy = permutedims(gy, [1,2,4,3])
    _gy = reshape(gy, size(gy,1)*size(gy,2)*size(gy,3), size(gy,4))
    _w = reshape(w, size(work,2), size(w,4))
    _gw = reshape(gw, size(_w,1), size(_w,2))
    gwork = zeros(work)
    BLAS.gemm!('N', 'T', T(1), _gy, _w, T(1), gwork)
    BLAS.gemm!('T', 'N', T(1), work, _gy, T(1), _gw)
    ccall(∇im2col_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        gwork, gx, size(gx,1), size(gx,2), size(gx,3), size(gx,4),
        size(w,1), size(w,2), padding[1], padding[2], strides[1], strides[2])
end

function convolution_mkl{T}(x::Array{T,4}, w::Array{T,4}, padding::NTuple{2,Int}, strides::NTuple{2,Int})
    y = MKL.convolution(x, w, padding, strides)
    function df(gy::Array)
        gx = MKL.∇convolution_data(x.data, w.data, gy, padding, strides)
        broadcast!(+, x.grad, x.grad, gx)
        gw = MKL.∇convolution_filter(x.data, w.data, gy, padding, strides)
        broadcast!(+, w.grad, w.grad, gw)
    end
end
