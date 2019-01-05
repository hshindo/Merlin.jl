@generated function repeat_kernel(x::CuArray{T,N}, counts::NTuple{N,Int}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void repeat(Array<$Ct,$N> y, Array<$Ct,$N> x) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= y.length()) return;
        int ndidxs[$N];
        y.ndindex(ndidxs, idx);
        for (int i = 0; i < $N; i++) ndidxs[i] = ndidxs[i] % x.dims[i];
        y[idx] = x(ndidxs);
    }
    """)
    quote
        ysize = Array{Int}(undef, N)
        for i = 1:N
            @assert counts[i] > 0
            ysize[i] = size(x,i) * counts[i]
        end
        y = similar(x, ysize...)
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, y, x)
        y
    end
end
