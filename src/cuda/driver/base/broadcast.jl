import Base: broadcast, broadcast!

for op in (:+, :-, :*)
    @eval begin
        function broadcast!{T,N}(::typeof($op), y::AbstractCuArray{T,N},
            x1::AbstractCuArray{T,N}, x2::AbstractCuArray{T,N})
            op = $op
            t = ctype(T)
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
            f(y, x1, x2, dx=length(y))
            y
        end
    end
end
