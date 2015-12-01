module Merlin

abstract Functor
abstract Optimizer

export Variable, diff!
export Concat, CrossEntropy, Linear, Lookup, Pooling, ReLU, Sequence, Window1D
#export Graph, Sequence
#export Add, Mult
export AdaGrad, Adam, SGD, optimize!

include("native.jl")
include("variable.jl")

include("functors/concat.jl")
include("functors/crossentropy.jl")
include("functors/linear.jl")
include("functors/lookup.jl")
#include("functors/math.jl")
include("functors/pooling.jl")
include("functors/relu.jl")
include("functors/sequence.jl")
include("functors/window1d.jl")

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

if haskey(ENV, "USE_CUDA")
  include("cuda/cuda.jl")
end

end
