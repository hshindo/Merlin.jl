module Merlin

abstract Functor
abstract Optimizer
using Base.LinAlg.BLAS

export Variable, diff!
export malloc, free
export ReLU, Tanh, Sigmoid
export Concat
export CrossEntropy
export Linear
export Lookup
export MaxPool2D
export Window2D

export AdaGrad, Adam, SGD, optimize!

zerograd(g) = nothing

include("native.jl")
include("variable.jl")

include("functors/activation.jl")
include("functors/concat.jl")
include("functors/crossentropy.jl")
include("functors/linear2.jl")
include("functors/lookup.jl")
#include("functors/math.jl")
include("functors/maxpool2d.jl")
#include("functors/sequence.jl")
include("functors/window2d.jl")

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

if haskey(ENV, "USE_CUDA")
  include("cuda/cuda.jl")
end

end
