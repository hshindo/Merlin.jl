# Overview
Merlin.jl provides many primitive functions.

## Decoding

1. Create `Variable` from Array (CPU) or CudaArray (CUDA GPU).
1. Create `Functor`s.
1. Apply the functors to the variable.

```julia
T = Float32
x = Variable(rand(T,50,5))
f = Linear(T,50,30)
y = f(x)
```

## Training

1. Create `Optimizer`.
1. For each iteration...
  * Decode your variables.
  * Compute gradient.
  * Update `Functor`s with your `Optimizer`.

```julia
T = Float32
opt = SGD(0.001)

for i = 1:10
  x = Variable(rand(T,10,5))
  y = ReLU()(x)
  backward!(y)
  update!(opt, y)
end
```
