function (opt::SGD)(x::CuArray{T,N}, gx::CuArray{T,N}) where {T,N}
    if opt.momentum > 0.0
        throw("Not implemented yet.")
        if haskey(opt.states, x)
            v = opt.states[x]
        else
            v = zeros(x)
            opt.states[x] = v
        end
        m = T(opt.momentum)
        rate = T(opt.rate)
        v .= m .* v - rate * gx
        if opt.nesterov
            v = copy(v)
            BLAS.scal!(length(v), m, v, 1)
            BLAS.axpy!(-rate, gx, v)
        end
        BLAS.axpy!(T(1), v, x)
    else
        BLAS.axpy!(T(-opt.rate), gx, x)
    end
    fill!(gx, T(0))
end
