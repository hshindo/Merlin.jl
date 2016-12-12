export Conv

const IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32)
const ∇IM2COL_F32 = Libdl.dlsym(libmerlin, :im2col_f32_grad)

im2col_handle(::Type{Float32}) = IM2COL_F32
∇im2col_handle(::Type{Float32}) = ∇IM2COL_F32

type Conv{N} <: Functor
    w::Var
    window::Window{N}
end

function outsize{N}(w::Window{N}, x::Array)
    ntuple(i -> (size(x,i) + 2pad(w,i) - size(w,i)) ÷ stride(w,i) + 1, N)
end

"""
    Conv(T, filtersize, channels, pads, strides)

N-dimensional convolution function.

```julia
T = Float32
x = Var(rand(T,5,4,3,2))
f = Conv(T, Window(2,2,0,0,1,1), 3, 4)
y = f(x)
```
"""
function Conv{T,N}(::Type{T}, window::Window{N}, inchannel::Int, outchannel::Int)
    w = rand(T, window.dims..., inchannel, outchannel)
    #w = Array{T,N+2}(randn(window.dims..., inchannel, outchannel))
    w .*= 0.002
    w .-= 0.001
    Conv(zerograd(w), window)
end

function (f::Conv)(x::Var)
    x.data == nothing && return Var(nothing, f, (x,))
    setbackend!(f.w, typeof(x.data))
    y, df = conv(x, f.w, f.w.data, f.window, x.data)
    Var(y, f, (x,), df)
end

function conv{T}(vx::Var, vw::Var, w::Array{T,4}, win::Window{2}, x::Array{T,4})
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

function conv{T}(vx::Var, vw::Var, w::CuArray{T,4}, win::Window{2}, x::CuArray{T,4})
    y = CUDNN.convolution(x, w, win.pads, win.strides)
    function df(gy::CuArray)
        CUDNN.∇convolution_filter!(x, win.pads, win.strides, gy, vw.grad, beta=1.0)
        CUDNN.∇convolution_data!(w, win.pads, win.strides, gy, vx.grad, beta=1.0)
    end
    y, df
end

function ∇conv!{T}(gy::Array{T,4}, work::Array{T,3}, w::Array{T,4}, gw::Array{T,4}, win::Window{2}, x::Array{T,4}, gx::Array{T,4})
    gwork = zeros(work)
    w = reshape(w, size(work,2), size(w,4))
    gw = reshape(gw, size(w,1), size(w,2))
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
            size(win,1), size(win,2), pad(win,1), pad(win,2), stride(win,1), stride(win,2))
    end
end

function update!(f::Conv, opt)
    opt(f.w.data, f.w.grad)
end
