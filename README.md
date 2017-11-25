<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/Merlin.png" width="150"></p>

# Merlin: deep learning framework for Julia

`Merlin` is a deep learning framework written in [Julia](http://julialang.org/).

It aims to provide a fast, flexible and compact deep learning library for machine learning.

`Merlin` is tested against Julia `0.6` on Linux, OS X, and Windows (x64).

[![Build Status](https://travis-ci.org/hshindo/Merlin.jl.svg?branch=master)](https://travis-ci.org/hshindo/Merlin.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/ks18dkc3gucf0yso?svg=true)](https://ci.appveyor.com/project/hshindo/merlin-jl)

## Documentation
[![](https://img.shields.io/badge/docs-stable-blue.svg)](http://hshindo.github.io/Merlin.jl/stable/)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](http://hshindo.github.io/Merlin.jl/latest/)

## Requirements
- Julia 0.6
- g++ (for OSX or Linux)

## Installation
```julia
julia> Pkg.add("Merlin")
```

## Examples
* [MNIST](examples/mnist/)

## Quick Start
Basically,
1. Wrap your data with `Var` (Variable type).
2. Apply functions to `Var`.  
`Var` memorizes a history of function calls for auto-differentiation.
3. Compute gradients if necessary.
4. Update the parameters with an optimizer.

Here is an example of three-layer network:

<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/feedforward.png" width="120"></p>

`Merlin` supports both static and dynamic evaluation of neural networks.

### Dynamic Evaluation
```julia
using Merlin

T = Float32
x = zerograd(rand(T,10,5)) # instanciate Var with zero gradients
y = Linear(T,10,7)(x)
y = relu(y)
y = Linear(T,7,3)(y)

params = gradient!(y)
println(x.grad)

opt = SGD(0.01)
foreach(opt, params)
```
If you don't need gradients of `x`, use `x = Var(rand(T,10,5))` where `x.grad` is set to `nothing`.

### Static Evalation
For static evaluation, the process are as follows.
1. Construct a `Graph`.
2. Feed your data to the graph.

When you apply `Node` to a function, it's lazily evaluated.
```julia
using Merlin

T = Float32
x = Node()
y = Linear(T,10,7)(x)
y = relu(y)
y = Linear(T,7,3)(y)
@assert typeof(y) == Node
g = Graph(input=x, output=y)

x = zerograd(rand(T,10,10))
y = g(x)

params = gradient!(y)
println(x.grad)

opt = SGD(0.01)
foreach(opt, params)
```
When the network structure can be represented as *static*, it is recommended to use this style.

More examples can be found in [`examples`](examples/).
