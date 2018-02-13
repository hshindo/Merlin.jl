export window

function window1d(x::Matrix{T}, batchdims::Vector{Int}, ksize::Int, pad::Int, stride::Int, dilation::Int=1) where T
    y = zeros(T, size(x,1)*ksize, sum(batchdims))
    yi = 1
    i = 1
    n = size(x, 1)
    for dim in batchdims
        for i = -pad+cumdim:stride:pad+cumdim

        end
        i += dim
    end

    s = 1
    for dim in batchdims
        i = s - pad
        while i + ksize <= s + dim + pad
            for w = 0:ksize-1
                j = i + w * dilation
                if j >= s && j < s + dim
                    xi = (j-1) * n + 1
                    copy!(y, yi, x, xi, n)
                end
                yi += n
            end
            i += stride
        end
        s += dim
    end
    y
end
