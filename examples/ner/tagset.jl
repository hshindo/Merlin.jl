struct BIO
    dict::Dict
    tags::Vector
end
BIO() = BIO(Dict("O"=>1), ["O"])

Base.length(tagset::BIO) = length(tagset.tags)

function encode(tagset::BIO, tags::Vector{String})
    basetag = ""
    map(tags) do tag
        tag == "O" && return 1
        if tag == "_"
            tag = "I-" * basetag
        else
            basetag = tag
            tag = "B-" * basetag
        end
        get!(tagset.dict, tag) do
            id = length(tagset.dict) + 1
            push!(tagset.tags, tag)
            id
        end
    end
end

function decode(tagset::BIO, ids::Vector{Int})
    spans = Tuple{Int,Int,String}[]
    bpos = 0
    for i = 1:length(ids)
        tag = tagset.tags[ids[i]]
        tag == "O" && continue
        startswith(tag, "B-") && (bpos = i)
        nexttag = i == length(ids) ? "O" : tagset.tags[ids[i+1]]
        if !startswith(nexttag,"I-")
            basetag = tagset.tags[ids[bpos]][3:end]
            push!(spans, (bpos,i,basetag))
        end
    end
    spans
end
