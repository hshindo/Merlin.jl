export lookup

function lookup(w::Var, x::Var)
    configure!(w, x)
    y = lookup(w.data, x.data)
    Var(y, (lookup,w,x))
end

function lookup(w::Matrix{T}, x::Array{I}) where {T,I<:Integer}
    n = size(w, 1)
    y = zeros(T, n*size(x,1), Base.tail(size(x))...)
    for i = 1:length(x)
        x[i] <= 0 && continue
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        copy!(y, yi, w, wi, n)
    end
    y
end

function addgrad!(y::Var, ::typeof(lookup), w::Var, x::Var)
    isvoid(w.grad) && return
    ∇lookup!(y.grad, w.grad, x.data)
end

function ∇lookup!(gy::Array{T}, gw::Array{T}, x::Array{I}) where {T,I<:Integer}
    n = size(gw, 1)
    for i = 1:length(x)
        x[i] <= 0 && continue
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        py = pointer(gy, yi)
        pw = pointer(gw, wi)
        BLAS.axpy!(n, T(1), py, 1, pw, 1)
    end
end
