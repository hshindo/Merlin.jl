export Convolution

const IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32)
const ∇IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32_grad)

im2col_handle(::Type{Float32}) = IM2COL_F32
∇im2col_handle(::Type{Float32}) = ∇IM2COL_F32

@graph function im2col(x::Var, filtersize, padding, stride)
    y = im2col(x.data, filtersize, padding, stride)
    df(gy) = isconst(x) || ∇im2col!(x.grad, gy, filtersize, padding, stride)
    Var(y, [x], im2col, df)
end

function im2col{T}(x::Array{T,3}, y::Array{T,3}, filtersize::NTuple{2,Int}, padding::NTuple{2,Int}, stride::NTuple{2,Int})
    outdims = Int[(size(x,i)+2padding[i]-filtersize[i]) ÷ stride[i] + 1 for i=1:2]
    y = similar(x, outdims[1]*outdims[2], filtersize[1]*filtersize[2], size(x,3))
    h = im2col_handle(T)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        x, y, size(x,1), size(x,2), size(x,3)*size(x,4),
        filtersize[1], filtersize[2], padding[1], padding[2], stride[1], stride[2])
    y
end

function ∇im2col!{T}(gx::Array{T,4}, gy::Array{T,4}, filtersize::NTuple{2,Int}, padding::NTuple{2,Int}, stride::NTuple{2,Int})
    h = ∇im2col_handle(T)
    ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
        gy, gx, size(x,1), size(x,2), size(x,3)*size(x,4),
        filtersize[1], filtersize[2], padding[1], padding[2], stride[1], stride[2])
    gx
end

type Convolution{N} <: Functor
    w::Var
    padding::NTuple{N,Int}
    stride::NTuple{N,Int}
end

"""
    Convolution(T, filtersize, channelsize, padding, stride)

N-dimensional convolution function. Currently, only 2D is supported.

```julia
x = Var(rand(Float32,5,4,3,2))
f = Convolution(Float32, (2,2), (3,4), (0,0), (1,1))
y = f(x)
```
"""
function Convolution(T::Type, filtersize, channelsize, padding, stride)
    N = length(filtersize)
    w = rand(T, filtersize..., channelsize...)
    w .*= 0.002
    w .-= 0.001
    Convolution(Var(w), padding, stride)
end

@graph function (f::Convolution)(x::Var)
    if typeof(x.data) <: Array
        y, work = convolution(x.data, f.w.data, f.padding, f.stride)
    else
        y = convolution(x.data, f.w.data, f.padding, f.stride)
    end
    function df(gy)
        if typeof(x.data) <: Array
            ∇convolution!(x.data, x.grad, f.w.data, f.w.grad, f.padding, f.stride, work, gy)
        else
            CUDNN.∇convolution_filter!(x.data, gy, f.padding, f.stride, f.w.grad, beta=1.0)
            CUDNN.∇convolution_data!(f.w.data, gy, f.padding, f.stride, x.grad, beta=1.0)
        end
    end
    Var(y, [x], convolution, df)
end

function convolution()
end

function convolution{T}(x::Array{T,4}, w::Array{T,4}, padding::NTuple{2,Int}, stride::NTuple{2,Int})
    outdims = Int[(size(x,i)+2padding[i]-size(w,i)) ÷ stride[i] + 1 for i=1:2]
    work = Array(T, outdims[1]*outdims[2], size(w,1)*size(w,2)*size(x,3), size(x,4))
    _w = reshape(w, size(work,2), size(w,4))
    y = Array(T, size(work,1), size(_w,2), size(work,3))
    for i = 1:size(work,3)
        h = im2col_handle(T)
        _work = view(work, :, :, i)
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            x, _work, size(x,1), size(x,2), size(x,3),
            size(w,1), size(w,2), padding[1], padding[2], stride[1], stride[2])

        BLAS.gemm!('N', 'N', T(1), _work, _w, T(0), view(y,:,:,i))
    end
    y = reshape(y, outdims[1], outdims[2], size(y,2), size(y,3))
    y, work
end

convolution(x::CuArray, w::CuArray, padding, stride) = CUDNN.convolution(x, w, padding, stride)

function ∇convolution!{T}(x::Array{T,4}, gx::Array{T,4}, w::Array{T,4}, gw::Array{T,4}, padding, stride, work, gy)
    gwork = zeros(work)
    _w = reshape(w, size(work,2), size(w,4))
    _gw = reshape(gw, size(_w,1), size(_w,2))
    gy = reshape(gy, size(gy,1)*size(gy,2), size(gy,3), size(gy,4))
    for i = 1:size(work,3)
        _work = view(work, :, :, i)
        _gwork = view(gwork, :, :, i)
        _gy = view(gy, :, :, i)
        ∇gemm_A!('N', 'N', 1.0, _gwork, _w, _gy)
        ∇gemm_B!('N', 'N', 1.0, _work, _gw, _gy)

        h = ∇im2col_handle(T)
        _gx = view(gx, :, :, :, i)
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            _gwork, _gx, size(x,1), size(x,2), size(x,3),
            size(w,1), size(w,2), padding[1], padding[2], stride[1], stride[2])
    end
    gx
end

function ∇convolution!(x::CuArray, gx, w, gw, padding, stride, gy)
    CUDNN.∇convolution_filter!(x, gy, padding, stride, gw, beta=1.0)
    CUDNN.∇convolution_data!(w, gy, padding, stride, gx, beta=1.0)
end

function update!(f::Convolution, opt)
    opt(f.w.data, f.w.grad)
end
