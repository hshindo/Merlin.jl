<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

This is pre-alpha version. We will make it publicly available in a next few months.

[NLP Demo (temporary)](http://158.199.141.203/)

# Merlin.jl: deep learning framework in Julia

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).
It aims to provide a fast, flexible and compact deep learning library for machine learning.
Our primary goal is to develop a NLP toolkit based on `Merlin`.

`Merlin` is tested against Julia `0.4` and `nightly` on Linux, OS X, and Windows.

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/v2u1kyjy61ph0ihn/branch/master?svg=true)](https://ci.appveyor.com/project/hshindo/merlin-jl/branch/master)

## Requirements
- Julia 0.4 or later
- g++ (for OSX or Linux)

## Optional
- [cuDNN](https://developer.nvidia.com/cudnn) v4 (to use CUDA GPU)

## Installation
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
```
For OSX and Linux,
```julia
julia> Pkg.build("Merlin")
```
If you have installed mingw on Windows, you can build `Merlin` as well.

## Usage
- [Documentation (latest)](http://hshindo.github.io/Merlin.jl/latest/)

## Quick Start
### Decoding
```julia
using Merlin

x = Var(rand(Float32,10,5))
f = Network(
  Linear(Float32,10,7),
  ReLU(),
  Linear(Float32,7,3))
y = f(x)
```

### Training
```julia
using Merlin

data_x = [rand(Float32,10,5) for i=1:100] # input data
data_y = [Int[1,2,3] for i=1:100] # correct labels

f = Network(
  Linear(Float32,10,7),
  ReLU(),
  Linear(Float32,7,3))
lossfun = CrossEntropy()
opt = SGD(0.0001)

for epoch = 1:10
  println("epoch: $(epoch)")
  loss = fit(data_x, data_y, f, lossfun, opt)
  println("loss: $(loss)")
end
```

## Using CUDA
To be written...
