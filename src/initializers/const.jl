export Zeros, Ones

struct Zeros
end

(::Zeros){T}(::Type{T}, dims::Int...) = zeros(T, dims)

struct Ones
end

(::Ones){T}(::Type{T}, dims::Int...) = ones(T, dims)
