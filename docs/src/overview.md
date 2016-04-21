# Overview

## Decoding
1. Prepare data as `Array` (CPU) or `CudaArray` (CUDA GPU).
1. Create `Functor`s (function objects).
1. Apply the functors to your data.

``` julia
using Merlin

x = rand(Float32,50,5)
f1 = Linear(Float32,50,30)
f2 = ReLU()
y = x |> f1 |> f2 # or y = f2(f1(x))
```

## Training
1. Create `Optimizer`.
1. Decode your data.
1. Compute loss.
1. Compute gradient.
1. Update `Functor`s with the `Optimizer`.

``` julia
using Merlin

opt = SGD(0.001)
f = Graph(Linear(Float32,50,30), ReLU(), Linear(Float32,30,10)) # 3-layer network
train_data = [rand(Float32,50,1) for i=1:1000] # create 1000 training examples of size: (50,1)

for epoch = 1:10
  for i in randperm(length(train_data)) # shuffle
    x = train_data[i]
    y = f(x)
    label = [1] # assumes the correct label is always '1'
    loss = CrossEntropy(label)(y)
    gradient!(loss) # computes gradients of every parameters used in decoding
    update!(opt, f)
  end
end
```
