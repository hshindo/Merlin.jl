abstract Tagset

"""
    I: begin or internal, O: outside, E: end
"""
immutable BIO <: Tagset
    B::Int
    I::Int
    O::Int
end
BIO() = BIO(1,2,3)

function decode(tagset::BIO, tag::String)
    tag == "B" && return tagset.B
    tag == "I" && return tagset.I
    tag == "O" && return tagset.O
    throw("Invalid tag: $(tag)")
end

function decode(tagset::BIO, tags::Vector{Int})
    bpos = 0
    ranges = UnitRange{Int}[]
    for i = 1:length(tags)
        t = tags[i]
        next = i == length(tags) ? tagset.O : tags[i+1]
        t == tagset.B && (bpos = i)
        if (t == tagset.I && next != tagset.I) || (t == tagset.B && next != tagset.I)
            push!(ranges, bpos:i)
            bpos = 0
        end
    end
    ranges
end

#=
function encode(tagset::BIO, ranges::Vector{UnitRange{Int}})
    tags = fill(tagset.O, last(ranges[end]))
    for r in ranges
        tags[r] = tagset.I
        tags[last(r)] = tagset.E
    end
    tags
end
=#
