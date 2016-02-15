# Concat
Concatenates n-d arrays along the specified dimension.

## Params
- ***dim***::`Int` dimension

## Input
variable of n-d arrays. The number of dimensions of the input arrays must be the same.

## Output
variable of concatenated n-d array

## Example
```julia
> f = Concat(1)
> x1 = Variable(rand(Float32,10,5))
> x2 = Variable(rand(Float32,10,5))
> y = f(x1, x2)
```
