### Concat
Concatenates n-d arrays along the specified dimension.

#### Params
- dim::Int
dimension

#### Input
n-d arrays. The number of dimensions of the input arrays must be the same.

#### Output
concatenated n-d array

#### Example
```julia
f = Concat(1)
x = Var()
y = f(x)
```
