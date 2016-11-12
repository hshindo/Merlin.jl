type ArrayVar{T,N}
    data::Array{T,N}
    grad

end

function ArrayVar(dev::Int)

end
