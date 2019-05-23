export Identity

mutable struct Identity
end

(::Identity)(x) = x
inverse(::Identity, y) = y
