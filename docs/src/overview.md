# Overview
* Wrap data with `Var`.
* Apply functions to the variables.

```julia
x = Var(rand(Float32,10,5))

```

## Forward and Backward Computation
```@example
using Merlin #hide
x = param(rand(Float32,10,5))
f = Linear(Float32,10,7)
y = f(x)
gradient!(y)
```

## Training
Rather than manually call `gradient!`, Merlin provides `fit` function for training model.
```julia
using Merlin

data_x = [Var(rand(Float32,10,5)) for i=1:100] # input data
data_y = [Var([1,2,3]) for i=1:100] # correct labels

opt = SGD(0.0001)
for epoch = 1:10
  println("epoch: $(epoch)")
  loss = fit(f, crossentropy, opt, data_x, data_y)
  println("loss: $(loss)")
end
```
where `fit` tales five arguments: `decode`, `loss function`, `optimizer`.
