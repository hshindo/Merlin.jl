export Concat

type Concat <: Layer
    dim::Int
    xs::Vector{Layer}
end
Concat(dim::Int, xs::Layer...) = Concat(dim, [xs...])
