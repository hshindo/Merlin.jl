export Fill

doc"""
    Fill(x)

Fill initializer.
"""
struct Fill
    x
end

(f::Fill)(::Type{T}, dims::Int...) where T = fill(T(f.x), dims)
