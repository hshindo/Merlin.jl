export window1d, window1d_batch

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_float)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_grad_float)
window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

function window1d(x::Var, ksize, pad, stride, dilation)
    y = window1d(x.data, ksize, pad, stride, dilation)
    Var(y, window1d, (x,ksize,pad,stride,dilation))
end
window1d(x::Node, ksize, pad, stride, dilation) = Node(window1d, x, ksize, pad, stride, dilation)
window1d(x::Matrix, ksize, pad, stride, dilation) = window1d_batch(x, [size(x,2)], ksize, pad, stride, dilation)

function window1d_batch(x::Var, batchdims::Vector{Int}, ksize, pad, stride, dilation)
    y = window1d_batch(x.data, batchdims, ksize, pad, stride, dilation)
    Var(y, window1d_batch, (x,batchdims,ksize,pad,stride,dilation))
end
window1d_batch(x::Node, dims, ksize, pad, stride, dilation) = Node(window1d_batch, x, dims, ksize, pad, stride, dilation)
function window1d_batch{T}(x::Matrix{T}, batchdims, ksize, pad, stride, dilation)
    outdims = map(batchdims) do d
        (d + 2pad - ksize) ÷ stride + 1
    end
    y = similar(x, size(x,1)*ksize, sum(outdims))
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Ptr{Cint},Cint,Cint,Cint,Cint,Cint),
        x, y, size(x,1), map(Cint,outdims), length(outdims), ksize, pad, stride, dilation)
    y
end

function addgrad!(y::Var, ::typeof(window1d), x::Var, ksize, pad, stride, dilation)
    addgrad!(y, window1d_batch, x, [size(x.data,2)], ksize, pad, stride, dilation)
end

function addgrad!(y::Var, ::typeof(window1d_batch), x::Var, batchdims, ksize, pad, stride, dilation)
    isvoid(x.grad) && return
    ∇window1d_batch!(y.grad, x.grad, batchdims, ksize, pad, stride, dilation)
end

function ∇window1d_batch!{T}(gy::Array{T}, gx::Array{T}, batchdims, ksize, pad, stride, dilation)
    outdims = map(batchdims) do d
        (d + 2pad - ksize) ÷ stride + 1
    end
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Ptr{Cint},Cint,Cint,Cint,Cint,Cint),
        gy, gx, size(gx,1), map(Cint,outdims), length(outdims), ksize, pad, stride, dilation)
end
