type Embed
    w::SparseVar
end

"""
    Embed(path, T)

Construc embeddings from file.
"""
function Embed(path, T::Type)
    lines = open(readlines, path)
    ws = Array(Var, length(lines))
    for i = 1:length(lines)
        items = split(chomp(lines[i]), ' ')
        w = map(x -> parse(T,x), items)
        ws[i] = param(w)
    end
    Embed(ws, IntSet())
end
