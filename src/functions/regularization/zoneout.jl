export zoneout

function zoneout(x1::Var, x2::Var, rate::Float64)
    rate == 0.0 && return x2
    istraining() || return x2
    ydata, work = zoneout(x1.data, x2.data, rate)
    Var(ydata, ∇zoneout!, (x1,x2,rate,work))
end

function zoneout(x1::Array{T}, x2::Array{T}, rate::Float64) where T
    throw("Not implemented yet.")
end

@generated function zoneout(x1::CuArray{T}, x2::CuArray{T}, rate::Float64) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void zoneout($Ct *y, $Ct *x1, $Ct *x2, float *r, int n, float rate) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        y[idx] = r[idx] < rate ? x1[idx] : x2[idx];
    }""")
    quote
        @assert length(x1) == length(x2)
        y = similar(x1)
        r = curand(Float32, size(y))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(x1), pointer(x2), pointer(r), length(y), Float32(rate))
        y, r
    end
end

function ∇zoneout!(y::Var, x1::Var, x2::Var, rate::Float64, work)
    ∇zoneout!(y.grad, x1.grad, x2.grad, rate, work)
end

@generated function ∇zoneout!(gy::CuArray{T}, gx1::CuArray{T}, gx2::CuArray{T}, rate::Float64, r::CuArray{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void zoneout_grad($Ct *gy, $Ct *gx1, $Ct *gx2, float *r, int n, float rate) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        if (r[idx] < rate) gx1[idx] += gy[idx];
        else gx2[idx] += gy[idx];
    }""")
    quote
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), pointer(gx1), pointer(gx2), pointer(r), length(gy), Float32(rate))
    end
end
