# Overview
Merlin.jl provides many primitive functions.

## Decoding
1. Create `Variable` from Array (CPU) or CudaArray(CUDA GPU).
2. Create `Functor`s.
3. Apply the functors to the variable.

```julia
T = Float32
x = Variable(rand(T,50,5))
f = Linear(T,50,30)
y = f(x)
```

## Training
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
