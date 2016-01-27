type Token
  word::UTF8String
  wordid::Int
  charids::Vector{Int}
  catid::Int
end

function aaa(data::Vector)
  worddict = Dict{UTF8String,Int}()
  chardict = Dict{UTF8String,Int}()
  catdict = Dict{Any,Int}()

end

function read_conll(path)
  doc = []
  sent = []
  for line in open(readlines, path)
    line = chomp(line)
    if length(line) == 0
      push!(doc, sent)
      sent = []
    else
      items = split(line, '\t')
      push!(sent, tuple(items...))
    end
  end
  doc
end

function eval(golds::Vector{Token}, preds::Vector{Int})
  @assert length(golds) == length(preds)
  correct = 0
  total = 0
  for i = 1:length(golds)
    golds[i].catid == preds[i] && (correct += 1)
    total += 1
  end
  correct / total
end
