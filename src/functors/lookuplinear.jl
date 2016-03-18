type LookupLinear <: Functor
  f1::Lookup
  f2::Linear
  cache::Dict
end

function forward!(f::LookupLinear)
  f.f1
end

function lookuplinear()

end
