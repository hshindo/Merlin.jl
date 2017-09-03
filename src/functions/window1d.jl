export window1d

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_float)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_grad_float)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

function window1d(x::Var, winsize::Int, pad::Int, stride::Int, dilation::Int)
    batchdims = map(x.batchdims) do d
        (d + 2pad - winsize) ÷ stride + 1
    end
    data = similar(x.data, size(x.data,1)*winsize, sum(batchdims))
    window1d!(x.data, x.batchdims, data, winsize, pad, stride, dilation)
    Var(data, batchdims, window1d, (x,winsize,pad,stride,dilation))
end
function window1d(x::Node, winsize::Int, pad::Int, stride::Int, dilation::Int)
    Node(window1d, x, winsize, pad, stride, dilation)
end

function window1d!(x::Array{T,2}, batchdims, y::Array{T,2}, winsize, pad, stride, dilation) where {T}
    batchdims = map(Cint, batchdims)
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Ptr{Cint},Cint,Cint,Cint,Cint,Cint),
        x, y, size(x,1), batchdims, length(batchdims), winsize, pad, stride, dilation)
end

function addgrad!(y::Var, ::typeof(window1d), x::Var, winsize, pad, stride, dilation)
    isvoid(x.grad) && return
    ∇window1d!(y.grad, x.grad, x.batchdims, winsize, pad, stride, dilation)
end

function ∇window1d!(gy::Array{T}, gx::Array{T}, batchdims, winsize, pad, stride, dilation) where {T}
    batchdims = map(Cint, batchdims)
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Ptr{Cint},Cint,Cint,Cint,Cint,Cint),
        gy, gx, size(gx,1), batchdims, length(batchdims), winsize, pad, stride, dilation)
end
