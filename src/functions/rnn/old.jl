
@generated function transpose_batch(xs::Vector{CuMatrix{T}}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void transpose_batch(int n, $Ct *y, int *cumdimsY, $Ct **xs, int *cumdimsX) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n*cumdimsY[1]*cumdimsX[1]) return;

        int vj = idx / n;
        int vi = idx - vj * n;
        int j = vj / cumdimsY[1];
        int i = vj - j * cumdimsY[1];
        if (cumdimsY[j] + i < cumdimsY[j+1]) {
            int idxY = (cumdimsY[j] + i) * n + vi;
            int idxX = j * n + vi;
            y[idxY] = xs[i][idxX];
        }
    }""")
    quote
        batchsize_x = Array{Int}(length(xs))
        for i = 1:length(xs)
            batchsize_x[i] = size(xs[i], 2)
        end
        batchsize_y = transpose_dims(batchsize_x)
        cumdims_x = CuArray(cumsum_cint(batchsize_x))
        cumdims_y = CuArray(cumsum_cint(batchsize_y))
        y = CuArray{T}(size(xs[1],1), sum(batchsize_x))
        p_xs = CuArray(map(pointer,xs))
        gdims, bdims = cudims(size(xs[1],1)*batchsize_y[1]*batchsize_x[1])
        $k(gdims, bdims, size(xs[1],1), pointer(y), pointer(cumdims_y), pointer(p_xs), pointer(cumdims_x))
        y, batchsize_y
    end
end

@generated function transpose_batch2(x::CuMatrix{T}, batchsize_x::Vector{Int}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void transpose_batch(int n, $Ct *y, int *cumdimsY, $Ct *x, int *cumdimsX) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n*cumdimsY[1]*cumdimsX[1]) return;

        int vj = idx / n;
        int vi = idx - vj * n;
        int j = vj / cumdimsY[1];
        int i = vj - j * cumdimsY[1];
        if (cumdimsY[j] + i < cumdimsY[j+1]) {
            int idxY = (cumdimsY[j] + i) * n + vi;
            int idxX = (cumdimsX[i] + j) * n + vi;
            y[idxY] = x[idxX];
        }
    }""")
    quote
        batchsize_y = transpose_dims(batchsize_x)
        cumdims_x = CuArray(cumsum_cint(batchsize_x))
        cumdims_y = CuArray(cumsum_cint(batchsize_y))

        y = similar(x)
        gdims, bdims = cudims(size(x,1)*batchsize_y[1]*batchsize_x[1])
        $k(gdims, bdims, size(x,1), pointer(y), pointer(cumdims_y), pointer(x), pointer(cumdims_x))
        y, batchsize_y
    end
end
