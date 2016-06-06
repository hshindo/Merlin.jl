<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

This is alpha version.

[NLP Demo](http://158.199.141.203/)

# Merlin: deep learning framework in Julia

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).

It aims to provide a fast, flexible and compact deep learning library for machine learning.
Our primary goal is to develop a natural language processing toolkit based on `Merlin`.

`Merlin` is tested against Julia `0.4` and `nightly` on Linux, OS X, and Windows.

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/v2u1kyjy61ph0ihn/branch/master?svg=true)](https://ci.appveyor.com/project/hshindo/merlin-jl/branch/master)

## Requirements
- Julia 0.4 or later
- g++ (for OSX or Linux)

If you use CUDA GPU, the following is required.
- [cuDNN](https://developer.nvidia.com/cudnn) v5

## Installation
First, install [Julia](http://julialang.org/). Currently, version 0.4.x is recommended.

Then, clone the package from here:
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
```

For OSX and Linux, build `Merlin` as follows:
```julia
julia> Pkg.build("Merlin")
```
which generates `libmerlin.so` on `deps/`.

For Windows, `libmerlin.dll` is provided on `deps/`, however,
if you have installed `g++` with mingw-x64, you can build `Merlin.jl` on Windows.

To use CUDA GPU, install the following packages:
```julia
julia> Pkg.clone("https://github.com/hshindo/CUDA.jl.git")
julia> Pkg.clone("https://github.com/hshindo/CUDNN.jl.git")
```

## Quick Start
Basically,

1. Wrap your data with `Var`.
2. Apply functions to `Var`s. `Var` memorizes history of function application and use it for backpropagation.

`Merlin` supports two ways of writing network structure: *declarative* style and *imperative* style.

### Declarative Style
Static network can be defined by using `@graph` macro.
For example, a three-layer network can be constructed as follows:
```julia
using Merlin

f = @graph begin
  T = Float32
  x = Var(:x)
  x = Linear(T,10,7)(x)
  x = relu(x)
  x = Linear(T,7,3)(x)
  x
end
x = Var(rand(Float32,10,5)) # input variable
y = f(:x=>x) # output variable
```
where `Var(:<name>)` is a place-holder of input variable.

Similarly, GRU (gated recurrent unit) can be constructed as follows:
```julia
gru = @graph begin
  T = Float32
  xsize = 100
  ws = [param(rand(T,xsize,xsize)) for i=1:3]
  us = [param(rand(T,xsize,xsize)) for i=1:3]
  x = Var(:x)
  h = Var(:h)
  r = sigmoid(ws[1]*x + us[1]*h)
  z = sigmoid(ws[2]*x + us[2]*h)
  h_ = tanh(ws[3]*x + us[3]*(r.*h))
  h_next = (1 - z) .* h + z .* h_
  h_next
end
x = Var(rand(Float32,100,1))
h = Var(ones(Float32,100,1))
y = gru(:x=>x, :h=>h)
```

### Imperative Style
If the network structure is dependent on input data such as recurrent neural networks, imperative style is more intuitive.
```julia
using Merlin

T = Float32
f_h = Graph(...) # function for hidden unit
f_y = Graph(...) # function for output unit
h = Var(rand(T,100))

for i = 1:10
 x = data[i]
 h = f_h(concat(x,h))
 y[i] = f_out(h)
end
```

### Training
`Merlin` provides `fit` function to train your model.
```julia
using Merlin

data_x = [Var(rand(Float32,10,5)) for i=1:100] # input data
data_y = [Var([1,2,3]) for i=1:100] # correct labels

opt = SGD(0.0001)
for epoch = 1:10
  println("epoch: $(epoch)")
  loss = fit(f, crossentropy, opt, data_x, data_y)
  println("loss: $(loss)")
end
```
See documentation for more details.
