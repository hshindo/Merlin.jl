export window1d

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_float)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_grad_float)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

function window1d(x::Var, winsize::Int, pad::Int, stride::Int; dilation::Int=1)
    y = Var(nothing, window1d, (x,winsize,pad,stride,dilation))
    y.data = window1d(x.data, winsize, pad, stride, dilation)
    y.df! = () -> begin
        isvoid(x.grad) || ∇window1d!(y.grad, x.grad, winsize, pad, stride, dilation)
    end
    y
end

function window1d{T}(x::Array{T}, winsize::Int, pad::Int, stride::Int, dilation::Int)
    c = (length(x) + 2pad - winsize) ÷ stride + 1
    y = Array{T}(winsize, c)
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint),
        x, y, length(x), winsize, pad, stride, dilation)
    y
end

function ∇window1d!{T}(gy::Array{T}, gx::Array{T}, winsize::Int, pad::Int, stride::Int, dilation::Int)
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint,Cint),
        gy, gx, length(gx), winsize, pad, stride, dilation)
end
