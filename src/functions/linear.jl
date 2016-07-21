export Linear

type Linear
    w::Var
    b::Var
end

function Linear(T::Type, indim::Int, outdim::Int)
    r = T(sqrt(6 / (indim+outdim)))
    w = rand(-r, r, outdim, indim)
    b = fill(T(0), outdim, 1)
    Linear(Param(w), Param(b))
end

@compat (f::Linear)(x::Var) = linear(f.w, x, f.b)

function linear(w::Var, x::Var, b::Var)
    y = w.data * x.data
    broadcast!(.+, y, y, b.data)
    function df{T}(gy::UniArray{T})
        hasgrad(w) && BLAS.gemm!('N', 'T', T(1), gy, x.data, T(1), w.grad)
        hasgrad(x) && BLAS.gemm!('T', 'N', T(1), w.data, gy, T(1), x.grad)
        for offset = 1:length(b.data):length(gy)
            BLAS.axpy!(length(b.data), T(1), pointer(gy,offset), 1, pointer(b.grad), 1)
        end
    end
    Var(y, [w,x,b], df)
end
