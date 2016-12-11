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
    Conv(zerograd(w), pads, strides)
end

function (f::Conv)(x::Var)
    x.data == nothing && return Var(nothing, f, (x,))
    if typeof(x.data) <: Array
        y, work = f(x.data)
        df(gy::Array) = ∇conv!(f, gy, work, x.data, x.grad)
    elseif typeof(x.data) <: CuArray
        y = f(x.data)
        df(gy::CuArray) = ()
    else
        throw("Invalid data")
    end
    Var(y, f, (x,), df)
end

function (f::Conv{2}){T}(x::Array{T,4})
    work = Array{T}(outsize(f,x,1)*outsize(f,x,2), size(f.w,1)*size(f.w,2)*size(x,3), size(x,4))
    w::Array{T,2} = reshape(f.w.data, size(work,2), size(f.w,4))
    y = Array{T}(size(work,1), size(w,2), size(work,3))

    for i = 1:size(work,3)
        h = im2col_handle(T)
        work_i = view(work, :, :, i)
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            x, work_i, size(x,1), size(x,2), size(x,3),
            size(f.w,1), size(f.w,2), f.pads[1], f.pads[2], f.strides[1], f.strides[2])
        BLAS.gemm!('N', 'N', T(1), work_i, w, T(0), view(y,:,:,i))
    end
    y = reshape(y, outsize(f,x,1), outsize(f,x,2), size(y,2), size(y,3))
    y, work
end

function conv(x::CuArray, w::CuArray, padding, stride)
    y = CUDNN.convolution(x, w, padding, stride)
    function df(gy)
        CUDNN.∇convolution_filter!(x.data, gy, f.padding, f.stride, f.w.grad, beta=1.0)
        CUDNN.∇convolution_data!(f.w.data, gy, f.padding, f.stride, x.grad, beta=1.0)
    end
    y
end

function ∇conv!{T}(f::Conv{2}, gy::Array{T,4}, work::Array{T,3}, x::Array{T,4}, gx::Array{T,4})
    gwork = zeros(work)
    w::Array{T,2} = reshape(f.w.data, size(work,2), size(f.w,4))
    gw::Array{T,2} = reshape(f.w.grad, size(w,1), size(w,2))
    gy = reshape(gy, size(gy,1)*size(gy,2), size(gy,3), size(gy,4))

    for i = 1:size(work,3)
        work_i = view(work, :, :, i)
        gwork_i = view(gwork, :, :, i)
        gy_i = view(gy, :, :, i)
        ∇gemm_A!(gy_i, 'N', 'N', 1, gwork_i, w)
        ∇gemm_B!(gy_i, 'N', 'N', 1, work_i, gw)

        h = ∇im2col_handle(T)
        gx_i = view(gx, :, :, :, i)
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            gwork_i, gx_i, size(x,1), size(x,2), size(x,3),
            size(f.w,1), size(f.w,2), f.pads[1], f.pads[2], f.strides[1], f.strides[2])
    end
end

function ∇conv!(x::CuArray, gx, w, gw, padding, stride, gy)
    CUDNN.∇convolution_filter!(x, gy, padding, stride, gw, beta=1.0)
    CUDNN.∇convolution_data!(w, gy, padding, stride, gx, beta=1.0)
end

function update!(f::Conv, opt)
    opt(f.w.data, f.w.grad)
end
