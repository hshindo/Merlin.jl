# Functions

## Activation
```@docs
relu(x::Var)
sigmoid(x::Var)
tanh(x::Var)
```

```@docs
concat(dim::Int, xs::Vector{Var})

crossentropy(p, x::Var)
gemm(tA, tB, alpha, A::Var, B::Var)
max(x::Var, dim::Int)
reshape(x::Var, dims)
softmax(x::Var, dim::Int)
logsoftmax(x::Var, dim::Int)
sum(x, dim::Int)
transpose(x::Var)
```
