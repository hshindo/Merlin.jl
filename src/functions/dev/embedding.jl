struct EmbedVar <: AbstractVar
end

function lookup(x::Embedding)
end

function addgrad!(y::Var, ::typeof(lookup), x::Embedding)

end
