<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

This is alpha version.

# Merlin: deep learning framework in Julia

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).

It aims to provide a fast, flexible and compact deep learning library for machine learning.
Our primary goal is to develop a natural language processing toolkit based on `Merlin`.

`Merlin` is tested against Julia `0.5` on Linux, OS X, and Windows (x64).

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/v2u1kyjy61ph0ihn/branch/master?svg=true)](https://ci.appveyor.com/project/hshindo/merlin-jl/branch/master)

## Documentation
[![](https://img.shields.io/badge/docs-latest-blue.svg)](http://hshindo.github.io/Merlin.jl/latest/)

## Requirements
- Julia 0.5 or later
- g++ (for OSX or Linux)

## Installation
First, install [Julia](http://julialang.org/). Currently, version 0.5 is supported.

Then, clone and build the package.
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
julia> Pkg.build("Merlin")
```

## Quick Start
Basically,

1. Wrap your data with `Var` or `zerograd`.
2. Apply functions to `Var`. `Var` memorizes a history of functional applications for auto-differentiation.
3. Compute gradients if necessary.

Here is an example of three-layer network:

<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/feedforward.png" width="120"></p>

```julia
using Merlin

T = Float32
x = zerograd(rand(T,10,5))
y = Linear(T,10,7)(x)
y = relu(y)
y = Linear(T,7,3)(y)

gradient!(y)
println(x.grad)
```
If you don't need gradients of `x`, use `x = Var(rand(T,10,5))`.

When you apply `Var()` to a function, it's lazily evaluated.
```julia
T = Float32
x = Var()
y = Linear(T,10,7)(x)
y = relu(y)
y = Linear(T,7,3)(y)
@assert y.data == nothing

f = compile(y, x) # output: y, input: x
x = zerograd(rand(T,10,10))
y = f(x)
```
where `compile(y, x)` compiles the network structure from output variable: `y` and input variable: `x`, and create a `Graph` object.
When the network structure is *static*, it is recommended to use this style.

More examples can be found in [`examples`](examples/).

<!---
### Example2: Recurrent Neural Network (RNN)
<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/rnn.png" width="270"></p>

Dynamic network structures such as recurrent neural network (RNN) can be easily described with Julia's standard control-flow constructs (`for`, `if`, etc.).

```julia
using Merlin

T = Float32
f_h = @graph ... # function for hidden unit
f_y = @graph ... # function for output unit

h = Var(rand(T,50,1)) # initial hidden vector
xs = [constant(rand(T,50,1)) for i=1:10] # input data
ys = map(xs) do x
    c = concat(1, x, h) # concatanate x and h along the first dimension.
    h = f_h(c)
    f_y(h)
end
```
-->

### Training
Merlin provides a `fit` function to train your model.
```julia
train_x = [Var(rand(Float32,10,5)) for i=1:100] # input data
train_y = [[1,2,3] for i=1:100] # correct labels

f = begin
    T = Float32
    x = Var()
    y = Linear(T,10,7)(x)
    y = relu(y)
    y = Linear(T,7,3)(y)
    compile(y, x)
end

opt = SGD(0.0001)
for epoch = 1:10
  println("epoch: $(epoch)")
  loss = fit(train_x, train_y, f, crossentropy, opt)
  println("loss: $(loss)")
end
```

## Datasets
Common datasets are available via [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl).

## [Experimental] CUDA GPU
### Under Development...
If you use CUDA GPU, the following is required.
- [cuDNN](https://developer.nvidia.com/cudnn) v5 or later
- [JuCUDA.jl](https://github.com/hshindo/JuCUDA.jl.git) (CUDA bindings for Julia)

Install the following packages:
```julia
julia> Pkg.clone("https://github.com/hshindo/JuCUDA.jl.git")
```
