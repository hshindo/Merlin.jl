export BiAffine
export biaffine

struct BiAffine
    w::Var
    b::Var
end

function BiAffine(::Type{T}, insize::Int, outsize::Int; init_w=Xavier(), init_b=Zeros()) where T
    w = init_w(T, insize, outsize)
    b = init_b(T, outsize)
    BiAffine(zerograd(w), zerograd(b))
end

(f::BiAffine)(x) = biaffine(x, f.w, f.b)

function biaffine(x::Var, w::Var, b::Var)
    y = biaffine(x.data, w.data, b.data)
    Var(y, x.batchdims, biaffine, (x,w,b))
end
biaffine(x::Node, w::Var, b::Var; name="") = Node(biaffine, (x,w,b), name)

function biaffine(x1::Matrix, x2::Matrix, w::Matrix, b::Vector)
    #gemm('T', 'N', x1, )
    y = gemm('T', 'N', w, x)
    broadcast!(+, y, y, b)
end

function addgrad!(y::Var, ::typeof(linear), x::Var, w::Var, b::Var)
    isvoid(gx) || BLAS.gemm!('N', 'N', T(1), w, gy, T(1), gx)
    isvoid(gw) || BLAS.gemm!('N', 'T', T(1), x, gy, T(1), gw)
    isvoid(gb) || BLAS.axpy!(T(1), sum(gy,2), gb)
end
