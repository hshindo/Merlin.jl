export transpose_batch, transpose_batchdims

function transpose_batchdims(batchdims::Vector{Int})
    k = length(batchdims)
    t_batchdims = Int[]
    for t = 1:batchdims[1]
        while batchdims[k] < t
            k -= 1
        end
        push!(t_batchdims, k)
    end
    t_batchdims
end

@generated function transpose_batch(x::CuMatrix{T}, batchdims_x::Vector{Int}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void batch_rnn(int vecsize, $Ct *y, int *cumdimsY, int sizeY, $Ct *x, int *cumdimsX, int sizeX) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= vecsize*sizeX*sizeY) return;

        int vi = idx / vecsize;
        int voffset = idx - vi * vecsize;
        int j = vi / sizeY;
        int i = vi - j * sizeY;
        if (cumdimsY[j] + i < cumdimsY[j+1]) {
            int idxY = (cumdimsY[j] + i) * vecsize + voffset;
            int idxX = (cumdimsX[i] + j) * vecsize + voffset;
            y[idxY] = x[idxX];
        }
    }""")
    quote
        cumdims_x = Array{Cint}(length(batchdims_x)+1)
        cumdims_x[1] = 0
        for i = 2:length(cumdims_x)
            cumdims_x[i] = cumdims_x[i-1] + batchdims_x[i-1]
        end

        batchdims_y = transpose_batchdims(batchdims_x)
        cumdims_y = Array{Cint}(length(batchdims_y)+1)
        cumdims_y[1] = 0
        for i = 2:length(cumdims_y)
            cumdims_y[i] = cumdims_y[i-1] + batchdims_y[i-1]
        end

        y = similar(x)
        gdims, bdims = cudims(size(x,1)*batchdims_y[1]*batchdims_x[1])
        culaunch($f, gdims, bdims, size(x,1), y.ptr, CuArray(cumdims_y).ptr, batchdims_y[1],
            x.ptr, CuArray(cumdims_x).ptr, batchdims_x[1])
        y, batchdims_y
    end
end
