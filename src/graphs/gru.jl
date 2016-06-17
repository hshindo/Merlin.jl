export gru

"""
    GRU(::Type, xsize::Int)

Gated Recurrent Unit (GRU).
Ref: Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014

## Arguments
* xsize: size of input vector (= size of hidden vector)
"""
function gru{T}(::Type{T}, xsize::Int)
  @graph begin
    Ws = [param(rand(T,xsize,xsize)) for i=1:3]
    Us = [param(rand(T,xsize,xsize)) for i=1:3]
    x = Var(:x)
    h = Var(:h)
    r = sigmoid(Ws[1]*x + Us[1]*h)
    z = sigmoid(Ws[2]*x + Us[2]*h)
    h_ = tanh(Ws[3]*x + Us[3]*(r.*h))
    h_next = (1 - z) .* h + z .* h_
    h_next
  end
end
