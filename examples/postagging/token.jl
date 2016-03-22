type Token
  word::UTF8String
  wordid::Int
  charids::Vector{Int}
  tagid::Int
end

function read_conll(path, append::Bool, worddict::Dict, chardict::Dict, tagdict::Dict)
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
      word = replace(items[2], r"[0-9]", '0') |> UTF8String
      wordid = begin
        w = lowercase(word)
        get(worddict, w, unkword)
      end
      chars = convert(Vector{Char}, word)
      charids = map(chars) do c
        append ? get!(chardict, c, length(chardict)+1) : chardict[c]
      end
      tag = items[5]
      tagid = append ? get!(tagdict, tag, length(tagdict)+1) : tagdict[tag]
      token = Token(word, wordid, charids, tagid)
      push!(sent, token)
    end
  end
  doc
end

function read_wordlist(path)
  d = Dict()
  for l in open(readlines, path)
    get!(d, chomp(l), length(d)+1)
  end
  d
end

function make_batch(data::Vector{Vector{Token}})
  dict = Dict()
  for tokens in data
    key = length(tokens)
    haskey(dict, key) || (dict[key] = [])
    push!(dict[key], tokens)
  end
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
