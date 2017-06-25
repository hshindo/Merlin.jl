export window1d

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_float)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_grad_float)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

function window1d(x::Var, winsize::Int, pad::Int, stride::Int)
    y = Var(nothing, window1d, (x,winsize,pad,stride))
    isvoid(x.data) && return y

    y.data = window1d(x.data, winsize, pad, stride)
    y.df! = () -> begin
        isvoid(x.grad) || ∇window1d!(y.grad, x.grad, winsize, pad, stride)
    end
    y
end

function window1d(x::Array{T}, winsize::Int, pad::Int, stride::Int) where T
    c = (length(x) + 2pad - winsize) ÷ stride + 1
    y = Array{T}(winsize, c)
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        x, y, length(x), winsize, pad, stride)
    y
end

function ∇window1d!(gy::Array{T}, gx::Array{T}, winsize::Int, pad::Int, stride::Int) where T
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        gy, gx, length(gx), winsize, pad, stride)
end
