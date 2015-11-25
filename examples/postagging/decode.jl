function decode(data::Vector{Vector{Token}}, ls::LayerSet)
  golds, preds = Token[], Int[]
  for i = 1:length(data)
    toks = data[i]
    append!(golds, toks)
    node = forward(ls, toks)
    append!(preds, maxrows(node.data))
  end
  acc = eval(golds, preds)
  println("test acc: $(acc)")
end
