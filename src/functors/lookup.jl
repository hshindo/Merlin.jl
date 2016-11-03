type Lookup{T} <: Layer
    ws::Vector{Vector{T}}
    gws::Vector{Vector{T}}
    x::Layer
    y::Array{T}
end

tails(l::Lookup) = [l.x]

function Lookup()
end

function (f::Lookup)(y::Layer)
    
end

function forward!(l::Lookup)
    x = input(l)

end

function backward!(l::Lookup)
end
