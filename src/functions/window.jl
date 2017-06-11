export window1d, window2d

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_float)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_grad_float)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

function window1d(x::Var, insize::Int, pad::Int, stride::Int)
    data = window1d(x.data, insize, pad, stride)
    y = Var(data, window1d, (x,insize,pad,stride))
    y.df! = () -> begin
        isconst(x) || ∇window1d!(y.grad, x.grad, insize, pad, stride)
    end
    y
end

function window1d{T}(x::BatchedArray{T}, insize::Int, pad::Int, stride::Int)
    x = vec(x)
    ysize = Int[(x.size[i] + 2pad - insize) ÷ stride + 1 for i=1:batchsize(x)]
    y = BatchedArray(Array{T}(insize,sum(ysize)), ysize)
    xsize = map(Cint, x.size)
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{Cint},Cint,Ptr{T},Cint,Cint,Cint),
        x, xsize, batchsize(x), y, insize, pad, stride)
    y
end

function ∇window1d!{T}(gy::BatchedMatrix{T}, gx::BatchedArray{T}, insize::Int, pad::Int, stride::Int)
    gxsize = map(Cint, gx.size)
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint,Cint,Cint,Cint),
        gy, gx, gxsize, batchsize(gx), insize, pad, stride)
end
