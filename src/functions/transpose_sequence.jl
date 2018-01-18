@generated function transpose_sequence(x::CuMatrix{T}, batchdims_x::Vector{Int}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void batch_rnn($Ct *y, Array<$Ct,2> x, int *cumdims, int seqlength) {
        int idxX = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxX >= x.length()) return;

        int xj = idxX / x.dims[0];
        int xi = idxX - xj * x.dims[0];
        int j = xj / seqlength;
        int i = xj - j * seqlength;
        int yi = xi;
        int yj = cumdims[j] + i;
        if (yj < cumdims[j+1]) {
            int idxY = yi + yj * x.dims[0];
            y[idxY] = x[idx];
        }
    }""")
    quote
        k = length(batchdims_x)
        batchdims_y = Int[]
        for t = 1:batchdims_x[1]
            while batchdims_x[k] < t
                k -= 1
            end
            push!(batchdims_y, k)
        end

        y = CuArray{T}(length(batchdims_x)*size(x,1), batchdims_x[1])
        cumdims = Array{Cint}(size(y,2)+1)
        cumdims[1] = 0
        for i = 2:length(cumdims)
            cumdims[i] = cumdims[i-1] + batchdims_y[i-1]
        end

        gdims, bdims = cudims(length(x))
        culaunch($f, gdims, bdims, y.ptr, x, CuArray(cumdims).ptr, )
        y, batchdims_y
    end
end
