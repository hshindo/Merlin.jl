<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

# Merlin: deep learning framework for Julia

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).

It aims to provide a fast, flexible and compact deep learning library for machine learning.

`Merlin` is tested against Julia `1.0` on Linux, OS X, and Windows (x64).  
It runs on CPU and CUDA GPU.

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)

## Documentation
[![](https://img.shields.io/badge/docs-stable-blue.svg)](http://hshindo.github.io/Merlin.jl/stable/)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](http://hshindo.github.io/Merlin.jl/latest/)

## Requirements
- Julia 1.0

## Installation
The Pkg REPL-mode is entered from the Julia REPL using the key `]`.
```julia
(v1.0) pkg> add Merlin
```

## Quick Start
### MNIST
* See [MNIST](examples/mnist/)

More examples can be found in [`examples`](examples/).

### GPU Support
The following are required:
* [CUDA](https://developer.nvidia.com/cuda-toolkit) 9+
* [cuDNN](https://developer.nvidia.com/cudnn)
