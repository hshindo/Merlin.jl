module Merlin

# try
#    Pkg.installed("LibCUDA")
# catch
#    Pkg.clone("https://github.com/hshindo/LibCUDA.jl.git")
# end

using LibCUDA

mutable struct Config
    devname::String
    devids::Vector{Int}
    train::Bool
end
const CONFIG = Config("cpu", Int[], true)

iscpu() = CONFIG.devname == "cpu"
isgpu() = CONFIG.devname == "gpu"
function setdevice(devname::String, devids::Int...)
    CONFIG.devname = devname
    CONFIG.devids = [devids...]
end
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

include("functions/activation.jl")
include("cuda/functions/activation.jl")
include("functions/argmax.jl")
include("functions/blas.jl")
include("functions/concat.jl")
include("functions/conv.jl")
include("functions/dropout.jl")
include("functions/getindex.jl")
include("functions/linear.jl")
include("functions/lookup.jl")
include("functions/loss.jl")
include("functions/math.jl")
include("functions/pad.jl")
include("functions/reduce.jl")
include("functions/reshape.jl")
# include("functions/rnn.jl")
include("functions/softmax.jl")
include("functions/split.jl")
include("functions/standardize.jl")
include("functions/transpose_batch.jl")
include("functions/window1d.jl")

include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

end
