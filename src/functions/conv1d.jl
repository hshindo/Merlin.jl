export conv1d

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32_grad)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

"""
    conv1d(x::Var, dims, [pads, strides])

* x::Var: input var
* dims::Tuple: window size
* pads:Tuple: padding size
* strides::Tuple: stride size

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = conv1d(x, 10, 0, 1)
```
"""
conv1d(x::Var, ksize, pad, stride) = Conv1D(ksize,pad,stride)(x)

type Conv1D
    w::Var
    b::Var
    insize::Int
    outsize::Int
    pad::Int
    stride::Int
end

function Conv1D{T}(::Type{T}, insize::Int, outsize::Int, pad::Int, stride::Int)
    w = uniform(T, -0.001, 0.001, outsize, insize)
    b = zeros(T, outsize, 1)
    Conv1D(zerograd(w), zerograd(b), insize, outsize, pad, stride)
end

function (f::Conv1D)(x::Var)
    y = Var(nothing, f, (x,))
    conv1d!(y, x.data, f.insize, f.outsize, f.pad, f.stride)
    y
end

function conv1d!{T}(out::Var, x::Matrix{T}, insize::Int, outsize::Int, pad::Int, stride::Int)
    hlen = (length(x) + 2pad - insize) ÷ stride + 1
    h = Array{T}(insize, hlen)
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        x, h, length(x), insize, pad, stride)

    out.data = w * h .+ b
    out.df! = function df!()
        isvoid(out[1].grad) || ∇conv1d!(out.grad, out[1].grad, insize, pad, stride)
    end
end

function ∇conv1d!{T}(gy::Array{T}, gx::Array{T}, insize::Int, pad::Int, stride::Int)
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        gy, gx, length(gx), insize, pad, stride)
end
