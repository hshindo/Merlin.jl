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
    #using CUJulia
    #include("cuda/cudnn/CUDNN.jl")
    #using .CUDNN
    type CuArray{T,N}; end
    CuVector{T} = CuArray{T,1}
    CuMatrix{T} = CuArray{T,2}
else
    type CuArray{T,N}; end
    CuVector{T} = CuArray{T,1}
    CuMatrix{T} = CuArray{T,2}
end
UniArray{T,N} = Union{Array{T,N},CuArray{T,N}}
UniVector{T} = Union{Vector{T},CuVector{T}}
UniMatrix{T} = Union{Matrix{T},CuMatrix{T}}

abstract type Functor end

#include("batchedarray.jl")
include("util.jl")
include("var.jl")
include("graph.jl")
include("rand.jl")
#include("native.jl")
include("check.jl")
include("train.jl")

for name in [
    "activation",
    "argmax",
    "blas",
    "cat",
    "crossentropy",
    "dropout",
    "getindex",
    #"glu",
    "gru",
    "linear",
    "lookup",
    #"lstm",
    "math",
    "reduce",
    "normalize",
    "pairwise",
    #"pooling",
    "reshape",
    "softmax",
    "window",

    "conv1d",
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

include("hdf5.jl")
#include("caffe/Caffe.jl")

info("# threads: $(Threads.nthreads())")

end
