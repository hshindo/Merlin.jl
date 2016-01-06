module Merlin

abstract Optimizer
using Base.LinAlg.BLAS

abstract Functor

function call(f::Functor, x)
  f.x = x
  forward!(f)
  f.y
end

export Var, CudaVar, setvalue!, forward!, backward!
export clone
export Node
export Concat
export CrossEntropy
export Linear
export Lookup
export MaxPool2D
export ReLU
export Sigmoid
export Tanh
export Window2D

export Sequence

export AdaGrad, Adam, SGD, optimize!

include("native.jl")
include("abstractvar.jl")
include("var.jl")
include("node.jl")

#if haskey(ENV, "USE_CUDA")
include("cuda/CUDNN.jl")
include("cuda/cudavar.jl")
#end

include("functors2/concat.jl")
include("functors2/crossentropy.jl")
include("functors2/linear.jl")
include("functors2/lookup.jl")
#include("functors/math.jl")
include("functors2/maxpool2d.jl")
include("functors2/relu.jl")
#include("functors/sigmoid.jl")
#include("functors/tanh.jl")
#include("functors/sequence.jl")
include("functors2/window2d.jl")

include("sequence.jl")

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

end
