type Pooling{N} <: Var
    data
    grad
    tails::Vector
    windims::NTuple{N,Int}
    stride::NTuple{N,Int}
    paddims::NTuple{N,Int}
end

function maxpooling{T}(x::Array{T})
end

function meanpooling{T}(x::Array{T})
end

function ∇maxpooling!()
end
