type Token
  word
  wordid::Int
  charids::Vector{Int}
  catid::Int
end

function read_conll(path, append::Bool, worddict::Dict, chardict::Dict, catdict::Dict)
  doc = Vector{Token}[]
  sent = Token[]
  unkword = worddict["UNKNOWN"]
  for line in open(readlines, path)
    line = chomp(line)
    if length(line) == 0
      push!(doc, sent)
      sent = Token[]
    else
      items = split(line, '\t')
      word = replace(items[2], r"[0-9]", '0')
      wordid = begin
        w = lowercase(word)
        get(worddict, w, unkword)
        #append ? get!(worddict, w, length(worddict)+1) : get(worddict, w, unkword)
      end
      chars = convert(Vector{Char}, word)
      charids = map(chars) do c
        append ? get!(chardict, c, length(chardict)+1) : chardict[c]
      end
      cat = items[5]
      catid = append ? get!(catdict, cat, length(catdict)+1) : catdict[cat]
      token = Token(word, wordid, charids, catid)
      push!(sent, token)
    end
  end
  doc
end

function accuracy(golds::Vector{Int}, preds::Vector{Int})
  @assert length(golds) == length(preds)
  correct = 0
  total = 0
  for i = 1:length(golds)
    golds[i] == preds[i] && (correct += 1)
    total += 1
  end
  correct / total
end
