export clipnorm!, clipvalue!

function maxnorm(x::Var, threshold::Float64)
    norm = norm(x.data, p=2)
    if norm <= threshold
        data = 0
    else
        
    end
    Var
end

function clipnorm!(x::Array{T}, threshold) where {T}
    (ndims(x) == 1 || ndims(x) == 2) || throw("Not supported yet.")
    z = mapreducedim(v -> v*v, +, x, 1)
    for j = 1:length(z)
        norm = sqrt(z[j])
        z[j] = norm <= T(threshold) ? 1 : T(threshold)/norm
    end
    x .*= z
end

function clipvalue!{T}(x::Array{T}, value)
    value = T(value)
    @inbounds for i = 1:length(x)
        x[i] = min(max(x[i],-value), value)
    end
end
