# Types
There are three basic types:

* `Variable`
* `Functor`
* `Optimizer`

## Variable
`Variable` has `value` and `grad`.
```julia
#x = Variable(AFArray(Float32,10,5))
#x.value
#x.grad
```

## Functor
`Functor` is an abstract type of functors.
