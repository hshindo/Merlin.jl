module Merlin

using Compat
using Base.LinAlg.BLAS
import Compat.view

@compat if is_windows()
  const libname = joinpath(dirname(@__FILE__), "../deps/libmerlin.dll")
else
  const libname = joinpath(dirname(@__FILE__), "../deps/libmerlin.so")
end
const libmerlin = Libdl.dlopen(libname)

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

include("util.jl")
include("var.jl")
include("gradient.jl")
include("graph.jl")
#include("training.jl")
include("native.jl")
#include("serialize.jl")

for name in [
  "activation/relu",
  "activation/sigmoid",
  "activation/tanh",
  "math/plus",
  "math/times",
  "concat",
  #"conv",
  "embed",
  "data",
  "embed",
  "linear",
  "reshape",
  "softmax",
  "sum",
  "window"
  ]
  include("layers/$(name).jl")
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
