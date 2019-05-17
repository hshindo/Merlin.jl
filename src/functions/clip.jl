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

function clipvalue!(x::Array{T}, value) where T
    value = T(value)
    @inbounds for i = 1:length(x)
        x[i] = min(max(x[i],-value), value)
    end
end
