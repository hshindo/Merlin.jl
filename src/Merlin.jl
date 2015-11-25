module Merlin

abstract Functor
abstract Optimizer

export Variable
export Concat, CrossEntropy, Identity, Linear, Lookup, Pooling, ReLU, Window1D
export Graph, Map, sequencial
#export Add, Mult
export AdaGrad, Adam, SGD

include("native.jl")
include("variable.jl")
include("functors/concat.jl")
include("functors/crossentropy.jl")
include("functors/graph.jl")
include("functors/identity.jl")
include("functors/linear.jl")
include("functors/lookup.jl")
include("functors/map.jl")
#include("functors/math.jl")
include("functors/pooling.jl")
include("functors/relu.jl")
include("functors/window1d.jl")

include("optimizers/adagrad.jl")
include("optimizers/adam.jl")
include("optimizers/sgd.jl")

if haskey(ENV, "USE_CUDA")
  include("cuda/cuda.jl")
end

end
