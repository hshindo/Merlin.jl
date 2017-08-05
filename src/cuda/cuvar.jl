mutable struct CuVar <: AbstractVar
    data
    batchdims
    args
    grad
    work
end
