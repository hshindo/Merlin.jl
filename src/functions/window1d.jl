export window1d

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_float)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_grad_float)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

function window1d(x::Var, winsize::Int, pad::Int, stride::Int, dilation::Int)
    batchdims_x = begin
        s = 1
        for i = 1:ndims(x.data)-1
            s *= size(x.data,i)
        end
        map(d -> d*s, x.batchdims)
    end
    batchdims_y = map(batchdims_x) do d
        (d + 2pad - winsize) ÷ stride + 1
    end

    data = similar(x.data, winsize, sum(batchdims_y))
    window1d!(x.data, batchdims_x, data, winsize, pad, stride, dilation)
    Var(data, batchdims_y, window1d, (x,winsize,pad,stride,dilation), work=batchdims_x)
end
function window1d(x::Node, winsize::Int, pad::Int, stride::Int, dilation::Int)
    Node(window1d, x, winsize, pad, stride, dilation)
end

function window1d!(x::Array{T}, batchdims, y::Matrix{T}, winsize, pad, stride, dilation) where {T}
    batchdims = map(Cint, batchdims)
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint,Cint,Cint,Cint,Cint),
        x, y, batchdims, length(batchdims), winsize, pad, stride, dilation)
end

function addgrad!(y::Var, ::typeof(window1d), x::Var, winsize, pad, stride, dilation)
    isvoid(x.grad) && return
    ∇window1d!(y.grad, x.grad, y.work, winsize, pad, stride, dilation)
end

function ∇window1d!(gy::Array{T}, gx::Array{T}, batchdims, winsize, pad, stride, dilation) where {T}
    batchdims = map(Cint, batchdims)
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint,Cint,Cint,Cint,Cint),
        gy, gx, batchdims, length(batchdims), winsize, pad, stride, dilation)
end
