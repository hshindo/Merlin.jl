# Merlin.jl
[Merlin.jl](https://github.com/hshindo/Merlin.jl) is a neural network library in [Julia](http://julialang.org/).

It aims to provide a fast, flexible and concise neural network library for machine learning.

Merlin.jl uses our customized [arrayfire](http://arrayfire.com/) library for array computation
 on CPU / GPU / OpenCL backends through [ArrayFire.jl](https://github.com/hshindo/ArrayFire.jl).

## ★ Install
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
```

## ★ Optional Requirements
* [cuDNN](https://developer.nvidia.com/cudnn) v4 (for CUDA GPU)
