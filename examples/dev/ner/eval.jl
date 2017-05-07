function fscore(pred::Vector{String}, gold::Vector{String})
    function f(data::Vector{String})
        ranges = UnitRange{Int}[]
        b = 0
        for i = 1:length(data)
            data[i] == "O" && continue
            data[i] != "_" && (b = i)
            (i == length(data) || data[i+1] != "_") && push!(ranges, b:i)
        end
        ranges
    end
    rp, rg = f(pred), f(gold)
    c = length(intersect(Set(rp),Set(rg)))
    prec = c / length(rp)
    recall = c / length(rg)
    f1 = 2 * recall * prec / (recall + prec)
    prec, recall, f1
end
