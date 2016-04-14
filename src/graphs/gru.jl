export GRU

"""
Gated Recurrent Unit (GRU)
Ref: Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014
"""
function GRU{T}(::Type{T}, xsize::Int, hsize::Int)
  # parameters
  Ws = [Variable(rand(T,xsize,hsize)) for i=1:3]
  Us = [Variable(rand(T,xsize,hsize)) for i=1:3]
  # input
  x = Variable()
  h = Variable()

  r = Sigmoid()(Ws[1]*x + Us[1]*h)
  z = Sigmoid()(Ws[2]*x + Us[2]*h)
  h_ = Tanh()(Ws[3]*x + Us[3]*(r.*h))
  out = (1 - z) .* h + z .* h_
  Graph(out)
end
