module MKL

const libmkl = Libdl.find_library(["mkl_rt"])
isempty(libmkl) && throw("MKL library cannot be found.")

include("lib/libmkl_dnn.jl")
include("lib/libmkl_dnn_types.jl")

include("convolution.jl")
include("pooling.jl")
include("relu.jl")

function check_dnnerror(e)
    e == E_SUCCESS && return
    e == E_INCORRECT_INPUT_PARAMETER && throw("INCORRECT_INPUT_PARAMETER.")
    e == E_UNEXPECTED_NULL_POINTER && throw("UNEXPECTED_NULL_POINTER.")
    e == E_MEMORY_ERROR && throw("MEMORY_ERROR.")
    e == E_UNSUPPORTED_DIMENSION && throw("UNSUPPORTED_DIMENSION.")
    e == E_UNIMPLEMENTED && throw("UNIMPLEMENTED.")
end

function dnnLayoutCreate{T}(x::Array{T})
    p = Ptr{Cvoid}[0]
    csize = Csize_t[size(x)...]
    cstrides = Csize_t[strides(x)...]
    dnnLayoutCreate_F32(p, ndims(x), csize, cstrides)
    p[1]
end

function dnnPrimitiveAttributesCreate()
    p = Ptr{Cvoid}[0]
    dnnPrimitiveAttributesCreate_F32(p)
    p[1]
end

function dnnExecute(primitive, resources)
    dnnExecute_F32(primitive, resources)
end

function dnnPrimitiveAttributesDestroy(attributes)
    dnnPrimitiveAttributesDestroy_F32(attributes)
end

function dnnDelete(primitive)
    dnnDelete_F32(primitive)
end

end
