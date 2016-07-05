type SharedEmbeds
  w::Matrix{Float32}
  iddict::Vector{Vector{Int}}
end

function lookup(w::SharedEmbeds, x::Array{Int})

end

function update!(f::SharedEmbeds)

end

type Var1
  value
  f
  tail::Var
  grad
end
