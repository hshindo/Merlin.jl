# Overview
Merlin.jl provides many primitive functions.

## Decoding
1. Create `Variable` from Array (CPU) or CudaArray (CUDA GPU).
1. Create `Functor`s.
1. Apply the functors to the variable.

```julia
using Merlin

x = Variable(rand(Float32,50,5))
f = Linear(Float32,50,30)
y = f(x)
```

## Training
1. Create `Optimizer`.
1. Decode your variables.
1. Compute gradient.
1. Update `Functor`s with your `Optimizer`.
```julia
using Merlin

opt = SGD(0.001)
f = [Linear(Float32,50,30), ReLU(), Linear(Float32,30,10)]

for i = 1:10
  x = Variable(rand(Float32,50,20))
  y = f(x)
  gradient!(y)
  update!(opt, f)
end
```
