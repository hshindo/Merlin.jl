@Var(Pooling{N},
windims::NTuple{N,Int},
stride::NTuple{N,Int},
paddims::NTuple{N,Int})

function maxpooling{T}(x::Array{T})
end

function meanpooling{T}(x::Array{T})
end

function ∇maxpooling!()
end
