import Base: size, ndim, length
export equals

for f in (:size,:ndims,:length,:eltype)
    @eval begin
        $f(x::Var) = forward0($f, x::Var)
        forward(::typeof($f), x) = $f(x), nothing
    end
end

equals(x::Var, y) = forward0(equals, x, y)
equals(x, y::Var) = forward0(equals, x, y)
forward(::typeof(equals), x, y) = x == y, nothing
