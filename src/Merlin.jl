module Merlin

try
    Pkg.installed("LibCUDA")
catch
    Pkg.clone("https://github.com/hshindo/LibCUDA.jl.git")
end

using LibCUDA
const UniArray{T,N} = Union{Array{T,N},CuArray{T,N}}
const UniMatrix{T} = Union{Matrix{T},CuMatrix{T}}
const UniVector{T} = Union{Vector{T},CuVector{T}}

mutable struct Config
    train::Bool
    debug::Bool
end
const CONFIG = Config(true, false)

export session
function session(f::Function; train=true, debug=false)
    CONFIG.train = train
    CONFIG.debug = debug
    f()
end

include("var.jl")
include("graph.jl")
include("test.jl")
include("initializer.jl")
include("optimizer.jl")
include("iterators.jl")
include("backend.jl")

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

include("functions/activation.jl")
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
include("functions/rnn.jl")
include("functions/softmax.jl")
include("functions/split.jl")
include("functions/standardize.jl")
include("functions/transpose_batch.jl")
include("functions/window1d.jl")

include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

end
