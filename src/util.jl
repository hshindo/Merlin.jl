function uniform{T}(::Type{T}, a, b, dims::Tuple)
    a < b || throw("Invalid interval: [$a: $b]")
    r = rand(T, dims)
    r .*= T(b - a)
    r .+= T(a)
    r
end
uniform{T}(::Type{T}, a, b, dims::Int...) = uniform(T, a, b, dims)
