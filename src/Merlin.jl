module Merlin

abstract Functor
abstract Optimizer
using Base.LinAlg.BLAS

#export Var, constant, backward!
export Var, CudaVar, setvalue!, Node, forward!, backward!
export Concat
export CrossEntropy
export Linear
export Lookup
export MaxPool2D
export ReLU
export Sigmoid
export Tanh
export Window2D

export AdaGrad, Adam, SGD, optimize!

include("native.jl")
include("var.jl")

#if haskey(ENV, "USE_CUDA")
include("cuda/CUDNN.jl")
include("cuda/cudavar.jl")
#end

#include("functors/concat.jl")
#include("functors/crossentropy.jl")
#include("functors/linear2.jl")
#include("functors/lookup.jl")
#include("functors/math.jl")
#include("functors/maxpool2d.jl")
include("functors2/relu.jl")
#include("functors/sigmoid.jl")
#include("functors/tanh.jl")
#include("functors/sequence.jl")
#include("functors/window2d.jl")

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

end
