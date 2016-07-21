module Merlin

using Compat
using Base.LinAlg.BLAS
import Compat.view

@compat if is_windows()
  const libmerlin = Libdl.dlopen(joinpath(Pkg.dir("Merlin"),"deps/libmerlin.dll"))
else
  const libmerlin = Libdl.dlopen(joinpath(Pkg.dir("Merlin"),"deps/libmerlin.dll"))
end

USE_CUDA = false
if USE_CUDA
  using CUDA
  using CUDNN
else
  type CuArray{T,N}
  end
  typealias CuVector{T} CuArray{T,1}
  typealias CuMatrix{T} CuArray{T,2}
end

typealias UniArray{T,N} Union{Array{T,N},CuArray{T,N}}

include("util.jl")
include("var.jl")
#include("gradient.jl")
#include("graph.jl")
include("training.jl")
include("native.jl")
#include("serialize.jl")

for name in [
    "activation",
    "concat",
    "conv",
    "crossentropy",
    "embed",
    "gemm",
    "linear",
    "reshape",
    "softmax",
    "transpose",
    ]
    include("functions/$(name).jl")
end

#for name in [
#    "gru"]
#  include("graphs/$(name).jl")
#end

export update!
for name in [
    "adagrad",
    "adam",
    "sgd"]
  include("optimizers/$(name).jl")
end

#include("caffe/Caffe.jl")

end
