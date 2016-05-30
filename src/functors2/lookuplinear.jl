type LookupLinear <: Functor
  ws::Vector{Variable}
  l::Linear
  cache::Vector{Variable}
end

function aaa(f::LookupLinear, x::Matrix{Int})
  for j = 1:size(x,2)
    for i = 1:size(x,1)
      id = x[i, j]

    end
  end
  v |> [f.f1, f.f2]
end

