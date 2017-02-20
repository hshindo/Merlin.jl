import Base: broadcast, broadcast!

immutable CArray{T,N}
    ptr::Ptr{T}
    dims::N
end

CArray{T,N}(x::CuArray{T,N}) = CArray(pointer(x),box(size(x)))

@generated function broadcast!{T,N}(::typeof(.+), y::CuArray{T,N}, x1::CuArray{T,N}, x2::CuArray{T,N})
    f = CuFunction("""
    $array_h
    __global__ void f(Array<$T,$N> y, Array<$T,$N> x1, Array<$T,$N> x2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < y.length()) {
            int subs[$N];
            y.idx2sub(idx, subs);
            y[idx] = x1(subs) + x2(subs);
        }
    }""")
    quote
        $f(CArray(y), CArray(x1), CArray(x2), dx=length(y))
        y
    end
end

#=
for op in (:+, :-, :*)
    @eval begin
        function broadcast!{T,N}(::typeof($op), y::AbstractCudaArray{T,N},
            x1::AbstractCudaArray{T,N}, x2::AbstractCudaArray{T,N})
            op = $op
            t = ctype(T)
            f = @nvrtc """
            $array_h
            __global__ void f(Array<$t,$N> y, Array<$t,$N> x1, Array<$t,$N> x2) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < y.length()) {
                    __shared__ int cumdims[$N];
                    cumdims[0] = 1;
                    for (int i = 1; i < N; i++) cumdims[i] = cumdims[i-1] * dims[i-1];

                    int temp = idx;
                    for (int i = N-1; i >= 1; i--) {
                        int k = temp / cumdims[i];
                        subs[i] = k;
                        temp -= k * cumdims[i];
                    }
                    subs[0] = temp;
                    y.idx2sub(idx, subs);
                    y(subs) = x1(subs) $op x2(subs);
                }
            } """
            #=
            f = @nvrtc """
            $array_h
            __global__ void f(Array<$t,$N> y, Array<$t,$N> x1, Array<$t,$N> x2) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < y.length()) {
                    int subs[$N];
                    y.idx2sub(idx, subs);
                    y(subs) = x1(subs) $op x2(subs);
                }
            } """
            =#
            f(y, x1, x2, dx=length(y))
            y
        end
    end
end
=#
