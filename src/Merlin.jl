module Merlin

using Base.LinAlg.BLAS
using JLD2
using CuArrays

if is_windows()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.dll"))
elseif is_linux() || is_apple()
    const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.so"))
else
    throw("Unsupported OS.")
end

#include("hdf5.jl")
include("graph.jl")
include("var.jl")
#include("native.jl")
include("check.jl")
include("initializer.jl")
include("optimizer.jl")

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
include("functions/embeddings.jl")
include("functions/getindex.jl")
include("functions/linear.jl")
include("functions/loss.jl")
include("functions/math.jl")
include("functions/recurrent.jl")
include("functions/reduce.jl")
include("functions/reshape.jl")
include("functions/softmax.jl")
include("functions/split.jl")
include("functions/standardize.jl")

include("datasets/Datasets.jl")
#include("caffe/Caffe.jl")

#info("#Threads: $(Threads.nthreads())")

end
