# Overview

## 1. Types
Basic types:

* `Variable`
* `Functor`
* `Optimizer`

### 1. Variable
`Variable` has `value` and `grad`.
```julia
> x = Variable(Array(Float32,10,5))
> x.value
> x.grad
```

### Functor
`Functor` is an abstract type of functors.

## Training
```julia
opt = SGD(0.001)

for i = 1:10
  v = Variable()
  y = v |> f1 |> f2
  y.grad = ones()
  backward!(y)
  update!(opt, y)
end
```
