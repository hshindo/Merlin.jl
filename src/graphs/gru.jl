"""
Gated Recurrent Unit (GRU)
See Chung et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", 2014
"""
type GRU
  w
  r
  u
end

function GRU()
  z = Variable()
  w = Variable(rand(Float32,10,5))
  h = (1 - z) * h + z * hh
  z = w * x + U * h
end

function forward!(f::GRU, x)
  z = f.w * x + f.u * h
end
