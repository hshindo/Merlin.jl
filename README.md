<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

This is pre-alpha version. We will make it publicly available in a next few months.

[NLP Demo (temporary)](http://158.199.141.203/)

# Merlin.jl: deep learning framework in Julia

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).
It aims to provide a fast, flexible and compact deep learning library for machine learning.
Our primary goal is to develop a NLP toolkit based on `Merlin`.

`Merlin` is tested against Julia `0.4` on Linux, OS X, and Windows.

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/v2u1kyjy61ph0ihn/branch/master?svg=true)](https://ci.appveyor.com/project/hshindo/merlin-jl/branch/master)

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
- [Documentation (latest)](http://hshindo.github.io/Merlin.jl/latest/)

## Quick Start
### Decoding
```julia
x = rand(Float32,10,5)
f = Graph(
  Linear(Float32,10,7),
  ReLU(),
  Linear(Float32,7,3)
)
y = f(x)
```

### Training
```julia
To be written...
```

## Using CUDA
To be written...
