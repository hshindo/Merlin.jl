export clipnorm!, clipvalue!

function clipnorm!{T}(x::Array{T}, threshold)
    z = mapreducedim(v -> v*v, +, x, 1)
    for j = 1:length(z)
        norm = sqrt(z[j])
        z[j] = norm <= T(threshold) ? 1 : T(threshold)/norm
    end
    x .*= z
end

function clipvalue!{T}(x::Array{T}, value::T)
    @inbounds @simd for i = 1:length(x)
        x[i] = min(max(x[i],-value), value)
    end
end
