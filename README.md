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
n = Node(name="x")
n = Linear(T,10,7)(n)
n = relu(n)
n = Linear(T,7,3)(n)
@assert typeof(n) == Node
g = Graph(n)

x = zerograd(rand(T,10,10))
y = g("x"=>x)

params = gradient!(y)
println(x.grad)

opt = SGD(0.01)
foreach(opt, params)
```
When the network structure can be represented as *static*, it is recommended to use this style.

## Examples
### MNIST
* See [MNIST](examples/mnist/)

### LSTM
<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/lstm.png" width="300"><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/lstm_batch.png" width="300"></p>

This is an example of batched LSTM.
```julia
using Merlin

T = Float32
x1 = rand(T,20,3)
x2 = rand(T,20,2)
x3 = rand(T,20,5)
x = Var(cat(2,x1,x2,x3))
f = BiLSTM(T, 20, 20) # input size: 20, output size: 20
y = f(x, [3,2,5])
```

More examples can be found in [`examples`](examples/).
