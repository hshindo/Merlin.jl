export gumbel_softmax

function gumbel_softmax(x::Var, temp::Float64)
    y = log(x) + gumbel(size(x.data))
    y = softmax(y / temp)
    y
end

function gumbel{T}(::Type{T}, dims::Tuple)
    const eps = 1e-20
    map(rand(T,dims)) do u
        -log(-log(u+eps) + eps)
    end
end
gumbel{T}(::Type{T}, dims::Int...) = gumbel(T, dims)
