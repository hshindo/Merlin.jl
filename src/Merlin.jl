module Merlin

using Base.LinAlg.BLAS

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.dll"))
elseif is_linux() || is_apple()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.so"))
else
    throw("Unsupported OS.")
end

const usecuda = begin
    libname = is_windows() ? "nvcuda" : "libcuda"
    !isempty(Libdl.find_library([libname]))
end

if usecuda
    using CUJulia
    include("cuda/cudnn/CUDNN.jl")
    using .CUDNN
else
    type CuArray{T,N}; end
    typealias CuVector{T} CuArray{T,1}
    typealias CuMatrix{T} CuArray{T,2}
end
typealias UniArray{T,N} Union{Array{T,N},CuArray{T,N}}
typealias UniVector{T} Union{Vector{T},CuVector{T}}
typealias UniMatrix{T} Union{Matrix{T},CuMatrix{T}}

abstract Functor

include("check.jl")
include("util.jl")
include("var.jl")
include("graph.jl")
include("fit.jl")
include("rand.jl")
#include("native.jl")
include("hdf5.jl")

for name in [
    "activation",
    "argmax",
    "blas",
    "cat",
    #"conv",
    "crossentropy",
    "dropout",
    "getindex",
    "gru",
    "linear",
    "lookup",
    "lstm",
    "math",
    "normalize",
    "pairwise",
    #"pooling",
    "reduce",
    "reshape",
    "softmax",
    "window",
    ]
    include("functions/$(name).jl")
    cudafile = "cuda/functions/$(name).jl"
    isfile(joinpath(dirname(@__FILE__),cudafile)) && include(cudafile)
end

export update!
for name in [
    "adagrad",
    "adam",
    "clipping",
    "sgd"]
    include("optimizers/$(name).jl")
end

#include("caffe/Caffe.jl")

end
