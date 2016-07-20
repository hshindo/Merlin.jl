module Merlin

using Compat
using Base.LinAlg.BLAS
import Compat.view

@compat if is_windows()
  const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.dll"))
else
  const libmerlin = Libdl.dlopen(joinpath(dirname(@__FILE__),"../deps/libmerlin.so"))
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
include("gradient.jl")
include("graph.jl")
include("training.jl")
include("native.jl")
include("serialize.jl")

for name in [
  "activation/relu",
  "activation/sigmoid",
  "activation/tanh",
  "blas/gemm",
  "math/plus",
  "math/times",
  "concat",
  "conv",
  "crossentropy",
  "data",
  "embed",
  "linear",
  "max",
  "reshape",
  "softmax",
  "sum",
  "transpose"
  ]
  include("functions/$(name).jl")
end

for name in [
    "gru"]
  include("graphs/$(name).jl")
end

export update!
for name in [
    "adagrad",
    "adam",
    "sgd"]
  include("optimizers/$(name).jl")
end

#include("caffe/Caffe.jl")

end
