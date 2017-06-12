function reducedim{T,N}(f::Function, x::BatchedArray{T,N}, dim::Int)

end

function rr(dims, js::Int, je::Int)
    for i = 1:dims[1]
        @inbounds for k = 1:dims[3]
            maxv = x[sub2ind(dims,i,1,k)]
            maxj = 1
            @inbounds for j = js:je
                ind = sub2ind(dims, i, j, k)
                if x[ind] > maxv
                    maxv = x[ind]
                    maxj = j
                end
            end
            y[i,k] = maxv
            inds[i,k] = maxj
        end
    end
end
