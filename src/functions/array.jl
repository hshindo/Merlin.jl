import Base: size, ndims, length

for f in (:size,:ndims,:length,:eltype)
    @eval begin
        $f(x::Var) = forward0($f, x::Var)
        forward(::typeof($f), x) = $f(x), nothing
    end
end
