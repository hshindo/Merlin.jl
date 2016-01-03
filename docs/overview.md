# Overview
Basic types:

* Var
* Functor
* Node

## Var
`Var{T}` is a type of variable.
```julia
x = Var(Float32, 10, 5)
```

## Functor
`Functor` is an abstract type of functors.
A functor has two functions: `forward` and `backward`.
```julia
x = Var(Float32, 10, 5)
f = ReLU()
y = forward!(f, x)
y.grad = ...
backward!(f)
```

## Node
`Node` is used for constructing a computation graph of functors.
```julia
n = Node(f)
```
