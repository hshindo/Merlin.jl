for op in (:exp, :log)
    @eval begin
        function $op{T,N}(x::AbstractCuArray{T,N})
            op = $op
            y = CuArray{T}(size(x))
            t = ctype(T)
            f = @nvrtc """
            $array_h
            __global__ void f(Array<$t,$N> x, Array<$t,$N> y) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < y.length()) {
                    y(idx) = $op(x(idx));
                }
            } """
            f(x, y, dx=length(y))
            y
        end
    end
end

function ∇exp!{T}(y::CuArray{T}, gy::CuArray{T}, gx::CuArray{T})

    f()
end
