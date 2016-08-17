type QuantEmbedding <: Functor
    ws::Vector{Var}
end

@compat function (f::QuantEmbedding)(x::Var)

end

function quant(f::QuantEmbedding, x::Array)

end
