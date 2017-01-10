export pairwise

function pairwise(x1::Var, x2::Var)
    (isa(x1.data,Void) || isa(x2.data,Void)) && return Var(nothing, pairwise, (x1,x2))
    y = pairwise(x1.data, x2.data)
    df(gy) = isa(x1.grad, Void) || isa(x2.grad, Void) || ∇pairwise!(gy, x1.grad, x2.grad)
    Var(y, df, (x1,x2))
end

function pairwise{T}(x1::Matrix{T}, x2::Matrix{T})
    m1, n1 = size(x1)
    m2, n2 = size(x2)
    y = Array{T}(m1+m2, n1, n2)
    offset = 1
    for i = 1:n1
        for j = 1:n2
            copy!(y, offset, x1, (i-1)*m1+1, m1)
            offset += m1
            copy!(y, offset, x2, (j-1)*m2+1, m2)
            offset += m2
        end
    end
    y
end

function ∇pairwise!{T}(gy::Array{T,3}, gx1::Matrix{T}, gx2::Matrix{T})
    m1, n1 = size(gx1)
    m2, n2 = size(gx2)
    offset = 1
    for i = 1:n1
        for j = 1:n2
            add!(gx1, (i-1)*m1+1, gy, offset, m1)
            offset += m1
            add!(gx2, (j-1)*m2+1, gy, offset, m2)
            offset += m2
        end
    end
end

#=
function pairwise(x1::Var, x2::Var)
    (isvoid(x1.data) || isvoid(x2.data)) && return Var(nothing, pairwise, (x1,x2))
    y = pairwise(x1.data, x2.data)
    df(gy) = isvoid(x1.grad) || isvoid(x2.grad) || ∇pairwise!(gy, x1.grad, x2.grad)
    Var(y, df, (x1,x2))
end

function pairwise{T}(x1::Matrix{T}, x2::Matrix{T})
    y = Array{T}(size(x1,2), size(x2,2), size(x1,1)+size(x2,1))
    for j = 1:size(x2,2)
        for i = 1:size(x1,2)
            @inbounds @simd for k = 1:size(x1,1)
                y[i,j,k] = x1[k,i]
            end
            o = size(x1,1)
            @inbounds @simd for k = 1:size(x2,1)
                y[i,j,k+o] = x2[k,j]
            end
        end
    end
    y
end

function ∇pairwise!{T}(gy::Array{T,3}, gx1::Matrix{T}, gx2::Matrix{T})
    for j = 1:size(gx2,2)
        for i = 1:size(gx1,2)
            @inbounds @simd for k = 1:size(gx1,1)
                gx1[k,i] += gy[i,j,k]
            end
            o = size(gx1,1)
            @inbounds @simd for k = 1:size(gx2,1)
                gx2[k,j] += gy[i,j,k+o]
            end
        end
    end
end
=#
