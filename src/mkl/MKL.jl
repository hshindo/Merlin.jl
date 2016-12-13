module MKL

const libmkl = Libdl.find_library(["mkl_rt"])
isempty(libmkl) && throw("MKL library cannot be found.")

include("lib/libmkl_dnn.jl")
include("lib/libmkl_dnn_types.jl")

function check_dnnerror(e)
    e == E_SUCCESS && return
    e == E_INCORRECT_INPUT_PARAMETER && throw("INCORRECT_INPUT_PARAMETER.")
    e == E_UNEXPECTED_NULL_POINTER && throw("UNEXPECTED_NULL_POINTER.")
    e == E_MEMORY_ERROR && throw("MEMORY_ERROR.")
    e == E_UNSUPPORTED_DIMENSION && throw("UNSUPPORTED_DIMENSION.")
    e == E_UNIMPLEMENTED && throw("UNIMPLEMENTED.")
end

include("convolution.jl")
include("relu.jl")

end
