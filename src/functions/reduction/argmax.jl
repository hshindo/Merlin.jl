import Base.argmax

function argmax(x::Array, dim::Int)
    map(x -> x[1], argmax(z.data,dims=1))
end

function argmax(x::CuArray, dim::Int)
    findmax(x, dims=dim)[2]
end
