export concat

function concat{T,N}(xs::Vector{Array{T,N}})
    p = Ptr{Void}[0]
    attr = primitive_attributes()
    #[create_layout() for x in xs]
    #dnnConcatCreate_F32(p, attr, length(xs), )
end
