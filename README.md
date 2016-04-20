<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

This is pre-alpha version. We will make this publicly available in a next few months.

# Merlin.jl: Deep Learning Framework in Julia

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/github/hshindo/Merlin.jl?branch=master)](https://ci.appveyor.com/project/hshindo/merlin-jl/branch/master)

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).
It aims to provide a fast, flexible and compact deep learning library for machine learning.
Compared with [Mocha.jl](https://github.com/pluskid/Mocha.jl) and other deep learning frameworks, `Merlin` is designed to describe dynamic network structure (e.g. recurrent neural network) more clearly and concisely.

- [Documentation (latest)](http://hshindo.github.io/Merlin.jl/latest/)

## Requirements
- Julia 0.4, 0.5
- g++ (for OSX or Linux)

## Optional
- [cuDNN](https://developer.nvidia.com/cudnn) v4 (to use CUDA GPU)

## Installation
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
```

## Usage

### Decoding
1. Prepare data as `Array` (CPU) or `CudaArray` (CUDA GPU).
1. Create `Functor`s (function objects).
1. Apply the functors to your data.

``` julia
using Merlin

x = rand(Float32,50,5)
f1 = Linear(Float32,50,30)
f2 = ReLU()
y = x |> f1 |> f2 # or y = f2(f1(x))
```

### Training
1. Create `Optimizer`.
1. Decode your data.
1. Compute loss.
1. Compute gradient.
1. Update `Functor`s with the `Optimizer`.

``` julia
using Merlin

opt = SGD(0.001)
f = Graph(Linear(Float32,50,30), ReLU(), Linear(Float32,30,10)) # 3-layer network
train_data = [rand(Float32,50,1) for i=1:1000] # create 1000 training examples of size: (50,1)

for epoch = 1:10
  for i in randperm(length(train_data)) # shuffle
    x = train_data[i]
    y = f(x)
    label = [1] # assumes the correct label is always '1'
    loss = CrossEntropy(label)(y)
    gradient!(loss) # computes gradients of every parameters used in decoding
    update!(opt, f)
  end
end
```

## Using CUDA
To be written...
