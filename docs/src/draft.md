
### Dynamic Evaluation
```julia
using Merlin

T = Float32
x = parameter(rand(T,10,5)) # instanciate Var with zero gradients
y = Linear(T,10,7)(x)
y = relu(y)
y = Linear(T,7,3)(y)

params = gradient!(y)
println(x.grad)

opt = SGD(0.01)
foreach(opt, params)
```
If you don't need gradients of `x`, use `x = Var(rand(T,10,5))` where `x.grad` is set to `nothing`.

### Static Evalation
For static evaluation, the process are as follows.
1. Construct a `Graph`.
2. Feed your data to the graph.

When you apply `Node` to a function, it's lazily evaluated.
```julia
using Merlin

T = Float32
n = Node(name="x")
n = Linear(T,10,7)(n)
n = relu(n)
n = Linear(T,7,3)(n)
@assert typeof(n) == Node
g = Graph(n)

x = param(rand(T,10,10))
y = g("x"=>x)

params = gradient!(y)
println(x.grad)

opt = SGD(0.01)
foreach(opt, params)
```
When the network structure can be represented as *static*, it is recommended to use this style.

## Quick Start
Basically,
1. Wrap your data with `Var` (Variable type).
2. Apply functions to `Var`.  
`Var` memorizes a history of function calls for auto-differentiation.
3. Compute gradients if necessary.
4. Update the parameters with an optimizer.

Here is an example of three-layer network:

<p align="center"><img src="https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/feedforward.png" width="120"></p>

`Merlin` supports both static and dynamic evaluation of neural networks.
