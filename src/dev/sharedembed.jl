type BlockEmbed <: Var
  w::Matrix
  alpha::Matrix
  rho::Float64
  iddict::Vector{Vector{Int}}
end

@compat function (w::SharedEmbed)(x::Array{Int})
  
end

function update!(l::SharedEmbed, opt)

end
