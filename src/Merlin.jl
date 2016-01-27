module Merlin

abstract Functor
export Functor
abstract Optimizer
export Optimizer

using ArrayFire
using Base.LinAlg.BLAS

export Variable, forward!, backward!
export Concat
export CrossEntropy
export Linear
export LogSoftmax
export Lookup
export MaxPool2D
export ReLU
export Sigmoid
export Tanh
export Window2D

export AdaGrad, Adam, SGD, optimize!

include("native.jl")
include("variable.jl")

#if haskey(ENV, "USE_CUDA")
#include("cudnn/CUDNN.jl")
#include("cuda/cudavar.jl")
#end

include("functors/concat.jl")
include("functors/crossentropy.jl")
include("functors/linear.jl")
include("functors/logsoftmax.jl")
include("functors/lookup.jl")
#include("functors/math.jl")
include("functors/maxpool2d.jl")
include("functors/relu.jl")
#include("functors/sigmoid.jl")
include("functors/softmax.jl")
#include("functors/tanh.jl")
include("functors/window2d.jl")

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

end
