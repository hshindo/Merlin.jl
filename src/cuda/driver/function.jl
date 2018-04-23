export CuFunction

mutable struct CuFunction
    ptr::Ptr{Void}
    mod::CuModule # avoid CuModule gc-ed
end

function CuFunction(mod::CuModule, name::String)
    ref = Ref{Ptr{Void}}()
    @apicall :cuModuleGetFunction (Ptr{Ptr{Void}},Ptr{Void},Cstring) ref mod name
    CuFunction(ref[], mod)
end

function CuFunction(ptx::String)
    mod = CuModule(ptx)
    fnames = String[]
    for line in split(ptx,'\n')
        m = match(r".visible .entry (.+)\(", line) # find function name
        m == nothing && continue
        push!(fnames, String(m[1]))
    end
    length(fnames) > 1 && throw("Multiple functions are found.")
    CuFunction(mod, fnames[1])
end

Base.unsafe_convert(::Type{Ptr{Void}}, f::CuFunction) = f.ptr
