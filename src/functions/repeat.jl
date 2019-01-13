import Base.repeat

function repeat(x::Var, counts::Int...)
    for i = 2:ndims(x)
        @assert size(x,i) == 1
    end
    all(c -> c == 1, counts) && return x
    y = repeat(x.data, counts...)
    Var(y, ∇repeat!, (x,counts))
end

function ∇repeat!(y::Var, x::Var, counts)
    isnothing(x.grad) && return
    gy = reshape(y.grad, size(x,1), prod(counts))
    gy = sum(gy, dims=2)
    addto!(x.grad, gy)
end
