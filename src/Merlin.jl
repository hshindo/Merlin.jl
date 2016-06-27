module Merlin

using Compat
using Base.LinAlg.BLAS

@windows? begin
  const libname = "libmerlin.dll"
end : begin
  const libname = "libmerlin.so"
end

const libpath = abspath(joinpath(dirname(@__FILE__), "..", "deps", libname))

try
  const global library = Libdl.dlopen(libpath)
catch y
  println("ERROR: Could not load native extension at $libpath. Try `Pkg.build("Merlin.jl")` to compile native codes.")
  throw(y)
end

if haskey(ENV, "USE_CUDA") && ENV["USE_CUDA"]
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
include("training.jl")
include("native.jl")
include("serialize.jl")

for name in [
  "activation",
  "concat",
  "linear",
  "lookup",
  "math",
  "max",
  "pooling",
  "reshape",
  "softmax",
  "softmax_crossentropy",
  "sum",
  "window2d"
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

#include("caffe/Caffe.jl")

end
