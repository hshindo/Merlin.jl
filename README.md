<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

This is alpha version.

[NLP Demo](http://jukainlp.hshindo.com/) (See [JukaiNLP](https://github.com/hshindo/JukaiNLP.jl.git) for more details.)

# Merlin: deep learning framework in Julia

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).

It aims to provide a fast, flexible and compact deep learning library for machine learning.
Our primary goal is to develop a natural language processing toolkit based on `Merlin`.

`Merlin` is tested against Julia `0.5` on Linux, OS X, and Windows (x64).

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
<!-- [![Build status](https://ci.appveyor.com/api/projects/status/v2u1kyjy61ph0ihn/branch/master?svg=true)](https://ci.appveyor.com/project/hshindo/merlin-jl/branch/master) -->

## Documentation
[![](https://img.shields.io/badge/docs-latest-blue.svg)](http://hshindo.github.io/Merlin.jl/latest/)

## Requirements
- Julia 0.5 or later
- g++ (for OSX or Linux)

## Installation
First, install [Julia](http://julialang.org/). Currently, version 0.5 is recommended.

Then, clone the package.
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
julia> Pkg.update()
```

For OSX and Linux, build `Merlin` as follows:
```julia
julia> Pkg.build("Merlin")
```
which generates `libmerlin.so` on `deps/`.

For Windows, `libmerlin.dll` is provided on `deps/`, however,
if you have installed `g++` with mingw-x64, you can build `Merlin`.

## Quick Start
Basically,

1. Wrap your data with `Var`.
2. Apply functions to `Var`s. `Var` memorizes a history of functional applications for auto-differentiation.
3. Compute gradient if necessary.

```julia
using Merlin

x = Var(rand(Float32,10,5))
zerograd!(x)
f = Linear(Float32,10,7)
y = f(x)
gradient!(y)
```

### Example1: Feed-Forward Neural Network
Static network can be constructed by `@graph` macro.

Here is an example of three-layer network:

<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/feedforward.png" width="120"></p>

```julia
using Merlin

T = Float32
ls = [Linear(T,10,7), Linear(T,7,3)]
g = @graph begin
    x = ls[1](:x)
    x = relu(x)
    x = ls[2](x)
    x
end
f = compile(g, :x) # create a function from graph
x = Var(rand(Float32,10,5))
y = f(x)
```
where `:x` is a place-holder for input argument.

### Example2: Recurrent Neural Network (RNN)
<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/rnn.png" width="270"></p>

Dynamic network structures such as recurrent neural network (RNN) can be easily described with Julia's standard control-flow constructs (`for`, `if`, etc.).

```julia
using Merlin

T = Float32
f_h = @graph ... # function for hidden unit
f_y = @graph ... # function for output unit

h = Var(rand(T,50,1)) # initial hidden vector
xs = [Var(rand(T,50,1)) for i=1:10] # input vars
ys = Array(Var, length(xs)) # output vars

for i = 1:length(xs)
    x = xs[i]
    c = concat(1, x, h) # concatanate x and h along the first dimension.
    h = f_h(c)
    ys[i] = f_y(h)
end
```

### Training
`Merlin` provides `fit` function to train your model.
```julia
data_x = [Var(rand(Float32,10,5)) for i=1:100] # input data
data_y = [Var([1,2,3]) for i=1:100] # correct labels
f = ...

opt = SGD(0.0001, momentum=0.9)
for epoch = 1:10
  println("epoch: $(epoch)")
  loss = fit(f, crossentropy, opt, data_x, data_y)
  println("loss: $(loss)")
end
```

## [Experimental] CUDA GPU
If you use CUDA GPU, the following is required.
- [cuDNN](https://developer.nvidia.com/cudnn) v5 or later
- [JuCUDA.jl](https://github.com/hshindo/JuCUDA.jl.git) (CUDA bindings for Julia)
- [JuCUDNN.jl](https://github.com/hshindo/JuCUDNN.jl.git) (cuDNN wrapper for Julia)

Install the following packages:
```julia
julia> Pkg.clone("https://github.com/hshindo/JuCUDA.jl.git")
julia> Pkg.clone("https://github.com/hshindo/JuCUDNN.jl.git")
```
