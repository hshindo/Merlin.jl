type SharedEmbeds
  w::Vector{Float32}
  iddict::Matrix{Int}
end

function lookup(w::SharedEmbed, x::Array{Int})
  for id in x
    w.iddict[id]
  end
end
