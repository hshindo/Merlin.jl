import Base.argmax

function argmax(x::Var; dims::Int)
    _argmax(x.data, dims=dims)
end

function _argmax(x::Array; dims::Int)
    map(x -> x[1], argmax(x,dims=dims))
end

function _argmax(x::CuArray; dims::Int)
    findmax(x, dims=dims)[2]
end
