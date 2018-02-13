module Merlin

using Base.LinAlg.BLAS

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(@__DIR__,"../deps/libmerlin.dll"))
elseif is_linux() || is_apple()
    const libmerlin = Libdl.dlopen(joinpath(@__DIR__,"../deps/libmerlin.so"))
else
    throw("Unsupported OS.")
end

try
    Pkg.installed("LibCUDA")
catch
    Pkg.clone("https://github.com/hshindo/LibCUDA.jl.git")
end
using LibCUDA
const UniArray{T,N} = Union{Array{T,N},CuArray{T,N}}
const UniMatrix{T} = Union{Matrix{T},CuMatrix{T}}
const UniVector{T} = Union{Vector{T},CuVector{T}}

include("var.jl")
include("graph.jl")
#include("native.jl")
include("test.jl")
include("initializer.jl")
include("optimizer.jl")
include("iterators.jl")
include("backend.jl")

mutable struct Config
    train::Bool
    debug::Bool
end
const CONFIG = Config(true, false)

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

info("#Threads: $(Threads.nthreads())")

end
