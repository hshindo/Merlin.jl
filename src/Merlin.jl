module Merlin

using Compat
using Base.LinAlg.BLAS

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

abstract Layer
export Data

include("util.jl")
include("layers/gradient.jl")
include("layers/graph.jl")
#include("var.jl")
#include("gradient.jl")
#include("graph.jl")
#include("training.jl")
include("native.jl")
#include("serialize.jl")

include("layers/activation.jl")
include("layers/concat.jl")
include("layers/data.jl")
include("layers/linear.jl")

#=
for name in [
  "activation",
  "concat",
  "conv",
  "linear",
  "logsoftmax",
  "lookup",
  "math",
  "max",
  "reshape",
  "softmax",
  "softmax_crossentropy",
  "sum"
  ]
  include("functions/$(name).jl")
end

for name in [
    "gru"]
  include("graphs/$(name).jl")
end

for name in [
    "adagrad",
    "adam",
    "sgd"]
  include("optimizers/$(name).jl")
end
=#

#include("caffe/Caffe.jl")

end
