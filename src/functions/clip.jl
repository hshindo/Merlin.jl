export clipnorm!, clipvalue!

function clipnorm!(x::Array{T}, threshold) where T
    (ndims(x) == 1 || ndims(x) == 2) || throw("Not supported yet.")
    z = mapreducedim(v -> v*v, +, x, 1)
    for j = 1:length(z)
        norm = sqrt(z[j])
        z[j] = norm <= T(threshold) ? 1 : T(threshold)/norm
    end
    x .*= z
end

function clipnorm!(x::CuArray{T}, threshold) where T
    (ndims(x) == 1 || ndims(x) == 2) || throw("Not supported yet.")
    z = mapreducedim(v -> v*v, +, x, 1)
    for j = 1:length(z)
        norm = sqrt(z[j])
        z[j] = norm <= T(threshold) ? 1 : T(threshold)/norm
    end
    x .*= z
end

function clipvalue!(x::Array{T}, value::T) where T
    value = T(value)
    @inbounds for i = 1:length(x)
        x[i] = min(max(x[i],-value), value)
    end
end

@generated function clipvalue!(x::CuArray{T}, value::T) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void clip($Ct *x, $Ct value, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        x[idx] = min(max(x[idx],-value), value);
    }""")
    quote
        @assert value > T(0)
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(x), value, length(x)) # TODO: handle with int case
        x
    end
end
