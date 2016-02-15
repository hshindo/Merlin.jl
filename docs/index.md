# Merlin.jl
Merlin is a neural network library in Julia language.

## Install
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
```

## Optional
* [cuDNN](https://developer.nvidia.com/cudnn) v4 (for CUDA GPU)

## Basics
```julia
using Merlin
using ArrayFire

setbackend("cpu")
T = Float32
x = Variable(Array(T,50,10))
y = x |> Linear(T, 30, 50) |> ReLU()
```
