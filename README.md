<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

This is pre-alpha version. We will make it publicly available in a next few months.

[NLP Demo (temporary)](http://158.199.141.203/)

# Merlin: deep learning framework in Julia

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).

It aims to provide a fast, flexible and compact deep learning library for machine learning.
Our primary goal is to develop a natural language processing toolkit based on `Merlin`.

`Merlin` is tested against Julia `0.4` and `nightly` on Linux, OS X, and Windows.

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/v2u1kyjy61ph0ihn/branch/master?svg=true)](https://ci.appveyor.com/project/hshindo/merlin-jl/branch/master)

## Documentation
[![](https://img.shields.io/badge/docs-latest-blue.svg)](http://hshindo.github.io/Merlin.jl/latest/)

## Requirements
- Julia 0.4 or later
- g++ (for OSX or Linux)

If you use CUDA GPU, the following is required.
- [cuDNN](https://developer.nvidia.com/cudnn) v5

## Installation
First, install Julia. Version 0.4.x is recommended.
Then, clone the package from here:
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
```
For OSX and Linux,
```julia
julia> Pkg.build("Merlin")
```
which generates `libmerlin.so` in `deps/`.

For Windows, `libmerlin.dll` is already prepared on `deps/`, however,
if you have installed mingw(x64), you can build `Merlin.jl` as well.

To enable CUDA, install the following two packages:
```julia
julia> Pkg.clone("https://github.com/hshindo/CUDA.jl.git")
julia> Pkg.clone("https://github.com/hshindo/CUDNN.jl.git")
```

## Quick Start
### Static Network
A three-layer network can be defined as follows:
```julia
f = Graph(
  x = Var()
  x = Linear(Float32,10,7)(x)
  x = relu(x)
  x = Linear(Float32,7,3)(x)
  x
)
x = Var(rand(Float32,10,5))
y = f(x)
```

Similarly, GRU (gated recurrent unit) can be defined as follows.
```julia
gru = Graph(
  T = Float32
  xsize = 100
  ws = [param(rand(T,xsize,xsize)) for i=1:3]
  us = [param(rand(T,xsize,xsize)) for i=1:3]
  x = Var()
  h = Var()
  r = sigmoid(ws[1]*x + us[1]*h)
  z = sigmoid(ws[2]*x + us[2]*h)
  h_ = tanh(ws[3]*x + us[3]*(r.*h))
  h_next = (1 - z) .* h + z .* h_
  h_next
)
x = Var(rand(Float32,100))
y = gru(x)
```

### Dynamic Network
```julia

```

### Training
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
