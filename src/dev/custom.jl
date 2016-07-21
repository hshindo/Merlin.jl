function custom(x::Var, f::Function)
    
end

type Custom <: Var
    data
    grad
    tails::Vector
end
