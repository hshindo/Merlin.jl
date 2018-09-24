export Orthogonal
import LinearAlgebra: svd

doc"""
    Orthogonal([gain=1.0])

# References
* Saxe et al., [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120)
"""
struct Orthogonal
    gain::Float64
end
Orthogonal() = Orthogonal(1.0)

function (o::Orthogonal)(::Type{T}, dim1::Int, dim2::Int) where T
    a = randn(T, dim1, dim2)
    F = svd(a)
    # u, _, v = svd(a)
    q = size(F.U) == (dim1,dim2) ? F.U : F.Vt
    q .* T(o.gain)
end
