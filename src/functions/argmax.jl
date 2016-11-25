export argmax

function argmax(x::Array, dim::Int)
    _, index = findmax(x, dim)
    ind2sub(size(x), vec(index))[dim]
end

function argmax(x::CuArray, dim)
    throw("Not implemented yet.")
end
