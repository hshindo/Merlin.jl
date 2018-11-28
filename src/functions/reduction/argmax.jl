import Base.argmax

function argmax(x::Array, dim::Int)
    map(x -> x[1], argmax(x,dims=1))
end

function argmax(x::CuArray, dim::Int)
    findmax(x, dims=dim)[2]
end
