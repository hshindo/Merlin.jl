# Overview

## Var
```@docs
Var
```

## Forward and Backward Computation
```julia
x = param(rand(Float32,10,5))
f = Linear(Float32,10,7)
y = f(x)
gradient!(y)
x.grad
```

## Training
Rather than call `gradient!` manually, Merlin provides `fit` function for training your model.
```julia
using Merlin

data_x = [constant(rand(Float32,10,5)) for i=1:100] # input data
data_y = [constant([1,2,3]) for i=1:100] # correct labels

opt = SGD(0.0001)
for epoch = 1:10
  println("epoch: $(epoch)")
  loss = fit(f, crossentropy, opt, data_x, data_y)
  println("loss: $(loss)")
end
```
