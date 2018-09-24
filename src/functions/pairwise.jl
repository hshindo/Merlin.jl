export pairwise

function pairwise(x1::Var, x2::Var)
    @assert length(x1.batchdims) == length(x2.batchdims)
    @assert ndims(x1) == ndims(x2) == 2
    y = pairwise(x1.data, x1.batchdims, x2.data, x2.batchdims)
    batchdims = [x1.batchdims[i]*x2.batchdims[i] for i=1:length(x1.batchdims)]
    Var(y, batchdims, pairwise, (x1,x2))
end

function pairwise{T}(x1::Matrix{T}, batchdims1::Vector{Int}, x2::Matrix{T}, batchdims2::Vector{Int})
    cumdim1 = 0
    cumdim2 = 0
    m1 = size(x1, 1)
    m2 = size(x2, 1)
    n = 0
    for i = 1:length(batchdims1)
        n += batchdims1[i] * batchdims2[i]
    end

    y = Array{T}(m1+m2, n)
    yi = 1
    for k = 1:length(batchdims1)
        n1 = batchdims1[k]
        n2 = batchdims2[k]
        for i = cumdim1+1:cumdim1+n1
            for j = cumdim2+1:cumdim2+n2
                copyto!(y, yi, x1, (i-1)*m1+1, m1)
                yi += m1
                copyto!(y, yi, x2, (j-1)*m2+1, m2)
                yi += m2
            end
        end
        cumdim1 += n1
        cumdim2 += n2
    end
    y
end

function addgrad!(y::Var, ::typeof(pairwise), x1::Var, x2::Var)
    T = eltype(y.data)
    gx1 = isvoid(x1.grad) ? Array{T}(0,0) : x1.grad
    gx2 = isvoid(x2.grad) ? Array{T}(0,0) : x2.grad
    ∇pairwise!(y.grad, gx1, x1.batchdims, gx2, x2.batchdims)
end

function ∇pairwise!{T}(gy::Matrix{T}, gx1::Matrix{T}, batchdims1::Vector{Int}, gx2::Matrix{T}, batchdims2::Vector{Int})
    cumdim1 = 0
    cumdim2 = 0
    m1 = size(gx1, 1)
    m2 = size(gx2, 1)
    yi = 1
    for k = 1:length(batchdims1)
        n1 = batchdims1[k]
        n2 = batchdims2[k]
        for i = cumdim1+1:cumdim1+n1
            for j = cumdim2+1:cumdim2+n2
                isempty(gx1) || BLAS.axpy!(m1, T(1), pointer(gy,yi), 1, pointer(gx1,(i-1)*m1+1), 1)
                yi += m1
                isempty(gx2) || BLAS.axpy!(m2, T(1), pointer(gy,yi), 1, pointer(gx2,(j-1)*m2+1), 1)
                yi += m2
            end
        end
        cumdim1 += n1
        cumdim2 += n2
    end
end
