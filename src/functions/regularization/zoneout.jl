export zoneout

function zoneout(x::Var, droprate::Float64, training::Bool)
    droprate == 0.0 && return x
    training || return x
    ydata, work = zoneout(x.data, droprate)
    Var(ydata, ∇zoneout!, (x,droprate,work))
end

function zoneout(x::Array{T}, droprate::Float64) where T
    work = rand(T, length(x))
    scale = T(1 / (1-droprate))
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = work[i] <= droprate ? T(0) : scale*x[i]
    end
    y, work
end

dropout(x::CuArray, droprate) = CUDNN.dropout(x, droprate)

function ∇dropout!(y::Var, x::Var, droprate::Float64, work)
    isnothing(x.grad) && return
    ∇dropout!(y.grad, x.grad, droprate, work)
end

function ∇dropout!(gy::Array{T}, gx::Array{T}, droprate::Float64, work::Vector{T}) where T
    scale = T(1 / (1-droprate))
    @inbounds for i = 1:length(gx)
        gx[i] += work[i] <= droprate ? T(0) : scale*gy[i]
    end
end

∇dropout!(gy::CuArray, gx::CuArray, droprate, dropdesc) = CUDNN.∇dropout!(gy, gx, dropdesc)
