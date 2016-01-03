### Linear
$$y = Wx + b$$
where \(W\) is a weight matrix, \(b\) is a bias vector.

#### Params
- `w::Var{T,2}` or `CudaVar{T,2}`

- `b::Var{T,1}`

#### Input
n-d array

#### Output
n-d array

#### Example
```julia
f = Linear()
x = Var()
y = f(x)
```
