export Convolution, convolution

#const IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32)
#const ∇IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32_grad)

#im2col_handle(::Type{Float32}) = IM2COL_F32
#∇im2col_handle(::Type{Float32}) = ∇IM2COL_F32

"""
    Convolution(T, filtersize, channels, padding, strides)

N-dimensional convolution function.

```julia
T = Float32
x = Var(rand(T,5,4,3,2))
f = Convolution(T, (2,2), (3,4), (0,0), (1,1))
y = f(x)
```
"""
type Convolution{N}
    w::Var
    padding::NTuple{N,Int}
    strides::NTuple{N,Int}
end

function Convolution(T::Type, filtersize, channels, padding, strides)
    w = rand(T, filtersize..., channels...)
    w .*= 0.002
    w .-= 0.001
    Convolution(zerograd(w), padding, strides)
end

function (f::Convolution)(x::Var, desc)
    x.data == nothing && return Var(nothing, f, (x,))
    setbackend!(f.w, typeof(x.data))
    convolution(typeof(x.data), x, f.w, f.padding, f.strides)
end

function convolution{T<:Array}(::Type{T}, x::Var, w::Var)
    #desc = MKL.convolution_desc(x.data, w.data, padding, strides)
    y = MKL.convolution(x.data, w.data, padding, strides)
    function df(gy::Array)
        gx = MKL.∇convolution_data(x.data, w.data, gy, desc)
        broadcast!(+, x.grad, x.grad, gx)
        gw = MKL.∇convolution_filter(x.data, w.data, gy, desc)
        broadcast!(+, w.grad, w.grad, gw)
    end
    Var(y, convolution, (x,w), df)
end

function convolution{T<:CuArray}(::Type{T}, x::Var, w::Var, padding, strides)
    y = CUDNN.convolution(x.data, w.data, padding, strides)
    function df(gy::CuArray)
        CUDNN.∇convolution_filter!(x.data, padding, strides, gy, w.grad, beta=1.0)
        CUDNN.∇convolution_data!(w.data, padding, strides, gy, x.grad, beta=1.0)
    end
    Var(y, convolution, (x,w), df)
end

#=
function convolution{T}(w::Array{T,4}, win::Window{2}, x::Array{T,4})
    outdims = outsize(win, x)
    work = Array{T}(outdims[1]*outdims[2], size(win,1)*size(win,2)*size(x,3), size(x,4))
    w = reshape(w, size(work,2), size(w,4))
    y = Array{T}(size(work,1), size(w,2), size(work,3))

    for i = 1:size(work,3)
        h = im2col_handle(T)
        work_i = view(work, :, :, i)
        ccall(h, Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
            x, work_i, size(x,1), size(x,2), size(x,3),
            size(win,1), size(win,2), pad(win,1), pad(win,2), stride(win,1), stride(win,2))
        BLAS.gemm!('N', 'N', T(1), work_i, w, T(0), view(y,:,:,i))
    end
    y = reshape(y, outdims[1], outdims[2], size(y,2), size(y,3))
    df(gy::Array) = ∇conv!(gy, work, vw.data, vw.grad, win, vx.data, vx.grad)
    y, df
end
=#
