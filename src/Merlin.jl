module Merlin

abstract Functor
abstract Optimizer
export Functor
export Optimizer

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

include("native.jl")

if haskey(ENV, "USE_CUDA")
  include("cuda/array.jl")
else
  type CudaArray{T,N}
  end
end

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
