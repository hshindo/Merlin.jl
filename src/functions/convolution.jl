export Conv

const IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32)
const ∇IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32_grad)

im2col_handle(::Type{Float32}) = IM2COL_F32
∇im2col_handle(::Type{Float32}) = ∇IM2COL_F32

type Conv{N} <: Functor
    w::Var
    pads::NTuple{N,Int}
    strides::NTuple{N,Int}
end

channels{N}(c::Conv{N}) = size(c.w,N+1), size(c.w,N+2)
filtersize{N}(c::Conv{N}) = ntuple(i -> size(c.w,i), N)

function outsize{N}(c::Conv{N}, x::Array, i::Int)
    (size(x,i) + 2*c.pads[i] - size(c.w,i)) ÷ c.strides[i] + 1
end

"""
    Conv(T, filtersize, channels, pads, strides)

N-dimensional convolution function.

```julia
x = Var(rand(Float32,5,4,3,2))
f = Conv(Float32, (2,2), (3,4), (0,0), (1,1))
y = f(x)
```
"""
function Conv(T::Type, filtersize, channels, pads, strides)
    N = length(filtersize)
    w = rand(T, filtersize..., channels...)
    w .*= 0.002
    w .-= 0.001
    Conv(Var(w), pads, strides)
end

function (f::Conv)(x::Var)
    x.data == nothing && return Var(nothing, f, (x,))
    y, df = f(x.data)
    Var(y, f, (x,), df)
end

function (f::Conv{2}){T}(x::Array{T,4})
    w::Array{T,4} = f.w.data
    work = Array{T}(outsize(f,x,1)*outsize(f,x,2), size(w,1)*size(w,2)*size(x,3), size(x,4))
    w = reshape(w, size(work,2), size(w,4))
    y = Array{T}(size(work,1), size(w,2), size(work,3))

    for i = 1:size(work,3)
        h = im2col_handle(T)
        work_i = view(work, :, :, i)
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            x, work_i, size(x,1), size(x,2), size(x,3),
            filtersize(f)..., f.pads..., f.strides...)
        BLAS.gemm!('N', 'N', T(1), work_i, w, T(0), view(y,:,:,i))
    end
    y = reshape(y, outsize(f,x,1), outsize(f,x,2), size(y,2), size(y,3))
    df(gy::Array) = ∇conv!(f, y, gy, x, gx, work)
    y, df
end

function conv(x::CuArray, w::CuArray, padding, stride)
    y = CUDNN.convolution(x, w, padding, stride)
    function df(gy)
        CUDNN.∇convolution_filter!(x.data, gy, f.padding, f.stride, f.w.grad, beta=1.0)
        CUDNN.∇convolution_data!(f.w.data, gy, f.padding, f.stride, x.grad, beta=1.0)
    end
    y, df
end

function ∇conv!{T}(f::Conv{2}, gy::Array{T,4}, work::Array{T,4}, x::Array{T,4}, gx::Array{T,4})
    gwork = zeros(work)
    w::Array{T,2} = reshape(f.w.data, size(work,2), size(f.w,4))
    gw::Array{T,2} = reshape(f.w.grad, size(f.w,1), size(f.w,2))
    gy = reshape(gy, size(gy,1)*size(gy,2), size(gy,3), size(gy,4))

    for i = 1:size(work,3)
        work_i = view(work, :, :, i)
        gwork_i = view(gwork, :, :, i)
        gy_i = view(gy, :, :, i)
        ∇gemm_A!('N', 'N', 1.0, gwork_i, w, gy_i)
        ∇gemm_B!('N', 'N', 1.0, work_i, gw, gy_i)

        h = ∇im2col_handle(T)
        gx_i = view(gx, :, :, :, i)
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            gwork_i, gx_i, size(x,1), size(x,2), size(x,3),
            filtersize(f)..., f.pads..., f.strides...)
    end
end

function ∇conv!(x::CuArray, gx, w, gw, padding, stride, gy)
    CUDNN.∇convolution_filter!(x, gy, padding, stride, gw, beta=1.0)
    CUDNN.∇convolution_data!(w, gy, padding, stride, gx, beta=1.0)
end

function update!(f::Convolution, opt)
    opt(f.w.data, f.w.grad)
end

function im2col(x::Var, filtersize, padding, stride)
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
