type Identity <: Functor
end

apply(fun::Identity, var::Variable) = var
