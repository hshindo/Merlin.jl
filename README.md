This is pre-alpha version. We will make this publicly available in a next few months.

# Merlin.jl: a deep learning framework in Julia

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/v2u1kyjy61ph0ihn/branch/master?svg=true)](https://ci.appveyor.com/project/hshindo/merlin-jl/branch/master)

`Merlin` is a deep learning library written in [Julia](http://julialang.org/).
It aims to provide a fast, flexible and minimal deep learning library for machine learning.
Compared with [Mocha.jl](https://github.com/pluskid/Mocha.jl) and other deep learning frameworks, `Merlin` is designed to describe dynamic network structure (e.g. recurrent neural network) more clearly and concisely.

- [Documentation (latest)](http://hshindo.github.io/Merlin.jl/latest/)

Some highlights:

- **Modular Architecture**: Merlin has a clean architecture with isolated components like network layers, activation functions, solvers, regularizers, initializers, etc. Built-in components are sufficient for typical deep (convolutional) neural network applications and more are being added in each release. All of them could be easily extended by adding custom sub-types.
- **High-level Interface**: Mocha is written in [Julia](http://julialang.org/), a high-level dynamic programming language designed for scientific computing. Combining with the expressive power of Julia and other its package eco-system, playing with deep neural networks in Mocha is easy and intuitive. See for example our IJulia Notebook example of [using a pre-trained imagenet model to do image classification](http://nbviewer.ipython.org/github/pluskid/Mocha.jl/blob/master/examples/ijulia/ilsvrc12/imagenet-classifier.ipynb).

## Requirements
- Julia 0.4
- g++ (for OSX or Linux)

## Optional
- [cuDNN](https://developer.nvidia.com/cudnn) v4 (to use CUDA GPU)

## Installation
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
```

## Usage

### Decoding
1. Create `Variable` from `Array` (CPU) or `CudaArray` (CUDA GPU).
1. Create `Functor`s.
1. Apply the functors to the variable.

``` julia
using Merlin

x = Variable(rand(Float32,50,5))
f = Linear(Float32,50,30)
y = f(x)
```

### Training
1. Create `Optimizer`.
1. Decode your variables.
1. Compute gradient.
1. Update `Functor`s with your `Optimizer`.

``` julia
using Merlin

opt = SGD(0.001)
f = [Linear(Float32,50,30), ReLU(), Linear(Float32,30,10)]

for i = 1:10
  x = Variable(rand(Float32,50,20))
  y = f(x) |> CrossEntropy(...)
  gradient!(y)
  update!(opt, f)
end
```

## Using CUDA
To be written...
