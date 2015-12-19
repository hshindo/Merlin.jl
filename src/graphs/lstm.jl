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

end

function forward(f::GRU, x)
  z = f.w * x + f.u * h
end
