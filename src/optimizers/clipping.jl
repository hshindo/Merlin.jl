export clipnorm!, clipvalue!

function clipnorm!{T}(x::Matrix{T}, threshold::T)
    z = mapreducedim(v -> v*v, +, x, 1)
    for j = 1:length(z)
        norm = sqrt(z[j])
        z[j] = norm <= threshold ? 1 : threshold/norm
    end
    x .*= z
end

function clipvalue!{T}(x::Array{T}, value::T)
    @inbounds @simd for i = 1:length(x)
        x[i] = min(max(x[i],-value), value)
    end
end
