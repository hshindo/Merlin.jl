module Merlin

using Compat
using Base.LinAlg.BLAS

#include("caffe/Caffe.jl")

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

"""
JIT compiler.
- `src`: source code
- `sym`: function name
"""
function cppcompile(src, sym::Symbol)
  dir = joinpath(dirname(@__FILE__), "..", "lib")
  symstr = string(sym)
  srcpath = joinpath(dir, "$(symstr).c")
  libname = @windows? "$(symstr).dll" : "$(symstr).so"
  libpath = joinpath(dir, libname)
  #Libdl.dlclose(eval(sym))

  compiler = "g++"
  open(srcpath, "w") do f
    write(f, src)
  end
  @windows? begin
    run(`$compiler -Wall -O3 -shared -o $libpath $srcpath`)
  end : begin
    run(`$compiler -fPIC -Wall -O3 -shared -o $libpath $srcpath`)
  end

  lib = Libdl.dlopen(libpath)
  h = Libdl.dlsym(lib, :run)
  @eval global $sym = $h
end

if haskey(ENV, "USE_CUDA")
  using CUDA
  using CUDNN
else
  type CuArray{T,N}
  end
  typealias CuVector{T} CuArray{T,1}
  typealias CuMatrix{T} CuArray{T,2}
end

include("util.jl")
export argmax
include("var.jl")
export Var, param, isparam, forward, gradient!
include("gradient.jl")
export approx_grad, checkgrad
include("graph.jl")
include("training.jl")

for name in [
  "activation",
  "concat",
  "conv",
  "crossentropy",
  "linear",
  "lookup",
  "math",
  "max",
  "reshape",
  "softmax",
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

end
