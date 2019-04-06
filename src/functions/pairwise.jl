export pairwise

"""
    pairwise(x1::Var, x2::Var)
"""
function pairwise(x::Var, dims::Vector{Int})
    indexes = Int[]
    off = 0
    for k = 1:length(dims)
        for j = 1:dims[k]
            for i = 1:dims[k]
                i == j && continue
                push!(indexes, off+i, off+j)
            end
        end
        off += dims[k]
    end
    indexes = reshape(indexes, 2, length(indexes)÷2)
    indexes = todevice(indexes)
    lookup(x, Var(indexes))
end

@generated function pairwise(x1::CuArray{T,3}, x2::CuArray{T,3}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void pairwise(Array<$Ct,4> y, Array<$Ct,3> x1, Array<$Ct,3> x2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= y.length()) return;

        int ndidxs[4];
        y.ndindex(ndidxs, idx);
        int m = x1.dims[0];
        if (ndidxs[0] < m) y[idx] = x1(ndidxs[0], ndidxs[1], ndidxs[3]);
        else y[idx] = x2(ndidxs[0]-m, ndidxs[2], ndidxs[3]);
    }""")
    quote
        @assert size(x1,3) == size(x2,3)
        y = similar(x1, size(x1,1)+size(x2,1), size(x1,2), size(x2,2), size(x1,3))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, y, x1, x2)
        y
    end
end

function pairwise(x1::Matrix{T}, batchdims1::Vector{Int}, x2::Matrix{T}, batchdims2::Vector{Int}) where T
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

function ∇pairwise!(y::Var, x1::Var, x2::Var)
    gx1 = isnothing(x1.grad) ? similar(x1.data) : x1.grad
    gx2 = isnothing(x2.grad) ? similar(x2.data) : x2.grad
    ∇pairwise!(y.grad, gx1, gx2)
end

function ∇pairwise!(gy::Matrix{T}, gx1::Matrix{T}, batchdims1::Vector{Int}, gx2::Matrix{T}, batchdims2::Vector{Int}) where T
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
