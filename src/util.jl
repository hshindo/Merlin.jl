function argmax(x, dim::Int)
  _, index = findmax(x, dim)
  ind2sub(size(x), vec(index))[dim]
end
