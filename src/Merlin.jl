module Merlin

using Base.LinAlg.BLAS
using JLD2

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(@__DIR__,"../deps/libmerlin.dll"))
elseif is_linux() || is_apple()
    const libmerlin = Libdl.dlopen(joinpath(@__DIR__,"../deps/libmerlin.so"))
else
    throw("Unsupported OS.")
end

using LibCUDA
const UniArray{T,N} = Union{Array{T,N},CuArray{T,N}}
const UniMatrix{T} = Union{Matrix{T},CuMatrix{T}}
const UniVector{T} = Union{Vector{T},CuVector{T}}

abstract type Functor end

include("var.jl")
include("backend.jl")
include("graph.jl")
#include("native.jl")
include("test.jl")
include("initializer.jl")
include("optimizer.jl")
include("iterators.jl")

for name in [
    "attention/add_attention",
    "pairwise",
    ]
    include("functions/$(name).jl")
    #isfile(joinpath(dirname(@__FILE__),cudafile)) && include(cudafile)
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
include("functions/reduce.jl")
include("functions/reshape.jl")
include("functions/rnn.jl")
include("functions/softmax.jl")
include("functions/split.jl")
include("functions/standardize.jl")
include("functions/transpose_batch.jl")

include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

#info("#Threads: $(Threads.nthreads())")

end
