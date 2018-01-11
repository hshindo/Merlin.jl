export lookup

function lookup(w::Var, x::Vector{Int})
    n = size(w, 1)
    y = similar(w.data, n, length(x))
    for i = 1:length(x)
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        copy!(y, yi, w.data, wi, n)
    end
    Var(y, (lookup,w,x))
end

function addgrad!(y::Var, ::typeof(lookup), w::Var, x::Vector{Int})
    isvoid(w.grad) && return
    ∇lookup!(y.grad, w.grad, x)
end

function ∇lookup!(gy::Array{T}, gw::Array{T}, x::Vector{Int}) where T
    n = size(gw, 1)
    for i = 1:length(x)
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        py = pointer(gy, yi)
        pw = pointer(gw, wi)
        BLAS.axpy!(n, T(1), py, 1, pw, 1)
    end
end

function ∇lookup!(gy::CuArray{T}, gw::CuArray{T}, x::Vector{Int}) where T
    n = size(gw, 1)
    for i = 1:length(x)
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        py = pointer(gy, yi)
        pw = pointer(gw, wi)
        BLAS.axpy!(n, T(1), py, 1, pw, 1)
    end
end
