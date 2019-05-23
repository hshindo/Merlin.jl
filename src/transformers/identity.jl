export Identity

mutable struct Identity
end

Identity(x) = Identity()
(::Identity)(x) = x
inverse(::Identity, y) = y
