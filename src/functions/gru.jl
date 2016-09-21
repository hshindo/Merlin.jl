export GRU

"""
    GRU(::Type, xsize::Int)

Gated Recurrent Unit (GRU).
See: Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014

## Args
* xsize: size of input vector (= size of hidden vector)

```julia
gru = GRU(Float32,100)
x = constant(rand(Float32,100))
h = Var(rand(Float32,100))
y = gru(x, h)
```
"""
function GRU(T::Type, xsize::Int)
    ws = [Var(rand(T,xsize,xsize)) for i=1:3]
    us = [Var(rand(T,xsize,xsize)) for i=1:3]
    x = GraphNode()
    h = GraphNode()
    r = sigmoid(ws[1]*x + us[1]*h)
    z = sigmoid(ws[2]*x + us[2]*h)
    h_ = tanh(ws[3]*x + us[3]*(r.*h))
    h_next = (1 - z) .* h + z .* h_
    compile(h_next)

    #=
    @graph begin
        x = :x
        h = :h
        r = sigmoid(ws[1]*x + us[1]*h)
        z = sigmoid(ws[2]*x + us[2]*h)
        h_ = tanh(ws[3]*x + us[3]*(r.*h))
        h_next = (1 - z) .* h + z .* h_
        h_next
    end
    =#
end

GRU_training(x::CuArray, hx::CuArray, cx::CuArray, droprate) =
    JuCUDNN.rnn_training(x, hx, cx, droprate, CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL, CUDNN_GRU)

GRU_inference(x::CuArray, hx::CuArray, cx::CuArray, w::CuArray, dropdesc) =
    JuCUDNN.rnn_inference(x, hx, cx, w, dropdesc, CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL, CUDNN_GRU)

∇GRU_data!(x::CuArray, gx::CuArray, hx::CuArray, ghx::CuArray, cx::CuArray,
    gcx::CuArray, w::CuArray, y::CuArray, gy::CuArray, ghy::CuArray,
    gcy::CuArray, dropdesc) = JuCUDNN.∇rnn_data!(
    x, hx, cx, w, y, gy, ghy, gcy, dropdesc, CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL, CUDNN_GRU, gx, ghx, gcx)

∇GRU_weight!(x::CuArray, hx::CuArray, w::CuArray, gw::CuArray, y::CuArray,
    dropdesc) = JuCUDNN.∇rnn_weight!(x, hx, y, w, dropdesc, CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL, CUDNN_GRU, gw)
