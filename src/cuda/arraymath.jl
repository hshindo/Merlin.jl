import Base: +, -, *, /

function +(x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    @assert length(x1) == length(x2)
    y = fill!(similar(x1), 0)
    axpy!(T(1), x1, y)
    axpy!(T(1), x2, y)
    y
end

function -(x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    @assert length(x1) == length(x2)
    y = fill!(similar(x1), 0)
    axpy!(T(1), x1, y)
    axpy!(T(-1), x2, y)
    y
end

function -(x::CuArray{T}) where T
    y = fill!(similar(x1), 0)
    axpy!(T(-1), x, y)
    y
end

*(A::CuMatrix{T}, x::CuVector{T}) where T = gemv('N', T(1), A, x)
*(A::CuMatrix{T}, B::CuMatrix{T}) where T = gemm('N', 'N', T(1), A, B)
