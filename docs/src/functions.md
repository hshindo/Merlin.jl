# Functions

```@docs
relu(x::Var)
sigmoid(x::Var)
tanh(x::Var)
concat(dim::Int, xs::Vector{Var})
Linear(w, b)
max(x::Var, dim::Int)
reshape(x::Var, dims)
softmax(x::Var)
logsoftmax(x::Var)
softmax_crossentropy(p::Var, x::Var, dim::Int)
sum(x::Var, dim::Int)
```
