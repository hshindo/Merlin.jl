type Token
  word::UTF8String
  wordid::Int
  charids::Vector{Int}
  catid::Int
end

function load_data(path)
  worddict = Dict{UTF8String,Int}("UNKNOWN"=>1)
  chardict = Dict{Char,Int}()
  catdict = Dict{ASCIIString,Int}()
  function to_token(word)
    word = UTF8String(word)
    wordid = begin
      w = replace(word, r"[0-9]", '0')
      get!(worddict, w, length(worddict)+1)
    end
    charids = map(convert(Vector{Char}, word)) do c
      get!(chardict, c, length(chardict)+1)
    end
    catid = get!(catdict, cat, length(catdict)+1)
    Token(word, wordid, charids, catid)
  end
  read_conll()
end

function read_conll(f, path, positions::Vector{Int})
  doc = []
  sent = []
  for line in open(readlines, path)
    line = chomp(line)
    if length(line) == 0
      push!(doc, sent)
      sent = []
    else
      items = map(split(line, '\t'))
      data = map(p -> items[p], positions)
      token = f(data)
      push!(sent, token)
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
