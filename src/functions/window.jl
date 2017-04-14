export window

const WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32)
const ∇WINDOW1D_F32 = Libdl.dlsym(libmerlin, :window1d_f32_grad)

window1d_handle(::Type{Float32}) = WINDOW1D_F32
∇window1d_handle(::Type{Float32}) = ∇WINDOW1D_F32

"""
    window(x::Var, dims, [pads, strides])

* x::Var: input var
* dims::Tuple: window size
* pads:Tuple: padding size
* strides::Tuple: stride size

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = window(x, (10,), pads=(0,), strides=(1,))
```
"""
window(x::Var, dims, pads, strides) = forward0(window, x, dims, pads, strides)

function window{N}(x::Var, dims::NTuple{N,Int}; pads=nothing, strides=nothing)
    pads == nothing && (pads = ntuple(_ -> 0, N))
    strides == nothing && (strides = ntuple(_ -> 1, N))
    window(x, dims, pads, strides)
end

function forward{T,N}(::typeof(window), x::Array{T},
    dims::NTuple{N,Int}, pads::NTuple{N,Int}, strides::NTuple{N,Int})
    c = (length(x) + 2pads[1] - dims[1]) ÷ strides[1] + 1
    y = Array{T}(dims[1], c)
    ccall(window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        x, y, length(x), dims[1], pads[1], strides[1])

    backward!(gy, gx) = isvoid(gx) || ∇window!(gy, gx, dims, pads, strides)
    y, backward!
end

@generated function forward{T,N}(::typeof(window), x::CuArray{T},
    dims::NTuple{N,Int}, pads::NTuple{N,Int}, strides::NTuple{N,Int})

    f = CuFunction("""
    __global__ void f($T *y, $T *x) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {

        }
    }
    """)
    quote

    end
end

function ∇window!{T}(gy::Array{T}, gx::Array{T}, dims::NTuple{1,Int}, pads, strides)
    ccall(∇window1d_handle(T), Void, (Ptr{T},Ptr{T},Cint,Cint,Cint,Cint),
        gy, gx, length(gx), dims[1], pads[1], strides[1])
end
