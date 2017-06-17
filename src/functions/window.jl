export window1d, window2d

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_float)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_grad_float)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

function window1d(x::Var, insize::Int, pad::Int, stride::Int)
    data, batchdims = window1d(x.data, x.batchdims, insize, pad, stride)
    y = Var(data, batchdims, window1d, (x,insize,pad,stride))
    y.df! = () -> begin
        isvoid(x.grad) || ∇window1d!(y.grad, x.grad, x.batchdims, insize, pad, stride)
    end
    y
end

function window1d(x::Array{T,N}, batchdims::Vector{Int}, insize::Int, pad::Int, stride::Int) where {T,N}
    n = Base.stride(x, N)
    batchdims = map(d -> d*n, batchdims)
    x = vec(x)
    ysize = map(batchdims) do d
        (d + 2pad - insize) ÷ stride + 1
    end
    y = Array{T}(insize, sum(ysize))
    xsize = map(Cint, batchdims)
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{Cint},Cint,Ptr{T},Cint,Cint,Cint),
        x, xsize, length(batchdims), y, insize, pad, stride)
    y, ysize
end

function ∇window1d!{T}(gy::Matrix{T}, gx::Array{T}, batchdims::Vector{Int}, insize::Int, pad::Int, stride::Int)
    n = Base.stride(gx, ndims(gx))
    gxsize = map(d -> Cint(d*n), batchdims)
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Ptr{Cint},Cint,Cint,Cint,Cint),
        gy, gx, gxsize, length(batchdims), insize, pad, stride)
end
