module Merlin

# try
#    Pkg.installed("LibCUDA")
# catch
#    Pkg.clone("https://github.com/hshindo/LibCUDA.jl.git")
# end

using LibCUDA

mutable struct Config
    devices::Vector{Int}
    train::Bool
end
const CONFIG = Config(Int[], true)

iscpu() = isempty(CONFIG.devices)
isgpu() = !iscpu()
istrain() = CONFIG.train
istrain(b::Bool) = CONFIG.train = b

include("var.jl")
include("graph.jl")
include("test.jl")
include("initializer.jl")
include("optimizer.jl")
include("iterators.jl")

#=
add!(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N} = broadcast!(+, y, y, x)
add!(x::CuArray{T,N}, y::CuArray{T,N}) where {T,N} = BLAS.axpy!(T(1), x, y)
@generated function add!(x::CuSubArray{T,N}, y::CuArray{T,N}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void add(Array<$Ct,$N> x, Array<$Ct,$N> y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= y.length()) return;
        y(idx) += x(idx);
    }""")
    quote
        @assert length(x) == length(y)
        gdims, bdims = cudims(length(x))
        culaunch($f, gdims, bdims, x, y)
        y
    end
end
@generated function add!(x::CuArray{T,N}, y::CuSubArray{T,N}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void add(Array<$Ct,$N> x, Array<$Ct,$N> y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= x.length()) return;
        y(idx) += x(idx);
    }""")
    quote
        @assert length(x) == length(y)
        gdims, bdims = cudims(length(x))
        culaunch($f, gdims, bdims, x, y)
        y
    end
end
=#

for name in [
    "activation",
    "argmax",
    "blas",
    "concat",
    "conv",
    "dropout",
    "getindex",
    "linear",
    "lookup",
    "loss",
    "math",
    "pad",
    "reduce",
    "reshape",
    "rnn",
    "softmax",
    "split",
    "standardize",
    "transpose_batch",
    "window1d"
    ]
    include("functions/$name.jl")
    isfile("cuda/functions/$name.jl") && include("cuda/functions/$name.jl")
end

include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

end
