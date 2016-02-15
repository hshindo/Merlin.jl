# CrossEntropy
Computes cross-entropy between a true distribution \(p\) and the target distribution \(q\).
$$\mathrm{H}(p,q)=-\sum_{x}p(x)\log q(x)$$

## Params
- ***p***::`Vector{Int}` or `Matrix{Float}`

## Input
- 2-d array

## Output
- 2-d array

## Example
```julia
p = [3, 7, 1]
f = CrossEntropy(p)
x = Variable()
y = f(x)
```
