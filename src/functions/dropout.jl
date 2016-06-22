dropout(x::Var, ratio::Float64) = Dropout(ratio)(x)

type Dropout
  ratio::Float64
end

@compat function (f::Dropout)(x::Var)
  @checkargs f (x,)
  throw("Not implemented yet.")
end
