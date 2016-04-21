<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

This is pre-alpha version. We will make this publicly available in a next few months.

# Merlin.jl: deep learning framework in Julia

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).
It aims to provide a fast, flexible and compact deep learning library for machine learning.
Compared with [Mocha.jl](https://github.com/pluskid/Mocha.jl) and other deep learning frameworks, `Merlin` is designed to describe dynamic network structure (e.g. recurrent neural network) more clearly and concisely.

`Merlin` is tested against Julia `0.4` and *current* `0.5-dev` on Linux, OS X, and Windows.

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/v2u1kyjy61ph0ihn/branch/master?svg=true)](https://ci.appveyor.com/project/hshindo/merlin-jl/branch/master)

## Requirements
- Julia 0.4 or 0.5-dev
- g++ (for OSX or Linux)

## Optional
- [cuDNN](https://developer.nvidia.com/cudnn) v4 (to use CUDA GPU)

## Installation
```julia
julia> Pkg.clone("https://github.com/hshindo/Merlin.jl.git")
```

## Usage

- [Documentation (latest)](http://hshindo.github.io/Merlin.jl/latest/)

## Using CUDA
To be written...
