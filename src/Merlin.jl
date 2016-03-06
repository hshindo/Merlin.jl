module Merlin

abstract Functor
abstract Optimizer
export Functor
export Optimizer

using Base.LinAlg.BLAS

export CudaArray
export Variable, forward!, backward!
export Concat
export CrossEntropy
export Linear
export LogSoftmax
export Lookup
export Max
export MaxPooling2D
export ReLU
export Reshape
export Tanh
export Window2D

export AdaGrad, Adam, SGD, update!

#use_cuda() = haskey(ENV, "USE_CUDA")

include("array.jl")
include("memory.jl")
include("native.jl")
include("util.jl")
include("variable.jl")

for name in ["concat",
             "crossentropy",
             "linear",
             "logsoftmax",
             "lookup",
             "max",
             "maxpooling2d",
             "relu",
             "reshape",
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
