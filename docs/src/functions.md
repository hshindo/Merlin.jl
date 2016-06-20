# Functions

```@docs
relu(x::Var)
sigmoid(x::Var)
tanh(x::Var)
concat(dim::Int, xs::Vector{Var})
crossentropy(p::Var, q::Var)
Linear(w, b)
max(x::Var, dim::Int)
reshape(x::Var, dims)
softmax(x::Var)
logsoftmax(x::Var)
sum(x::Var, dim::Int)
```
