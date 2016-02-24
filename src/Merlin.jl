module Merlin

abstract Functor
abstract Optimizer
export Functor
export Optimizer

#using ArrayFire
using Base.LinAlg.BLAS

export Variable, forward!, backward!
export Concat
export CrossEntropy
export Linear
export LogSoftmax
export Lookup
export MaxPooling
export ReLU
export Tanh
export Window2D

export AdaGrad, Adam, SGD, optimize!

# CUDNN
if haskey(ENV, "USE_CUDNN")
  #@windows? (
  #  begin
  #    const libcudnn = Libdl.find_library(["cudnn64_4"])
  #  end : begin
  #    const libcudnn = Libdl.find_library(["libcudnn"])
  #  end)
  #isempty(libcudnn) ? error("CUDNN library cannot be found.") : println("CUDNN is loaded.")

  using CUDNN
  #using CUDArt
end

include("native.jl")
include("util.jl")
include("variable.jl")

for name in ["concat",
             "crossentropy",
             "linear",
             "logsoftmax",
             "lookup",
             "maxpooling2d",
             "relu",
             "tanh",
             "window2d"]
  include("functors/$(name).jl")
end

for name in ["adagrad",
             "adam",
             "sgd"]
  include("optimizers/$(name).jl")
end

end
