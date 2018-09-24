export CuFunction

mutable struct CuFunction
    ptr::Ptr{Cvoid}
    mod::CuModule # avoid CuModule gc-ed
end

function CuFunction(mod::CuModule, name::String)
    ref = Ref{Ptr{Cvoid}}()
    @apicall :cuModuleGetFunction (Ptr{Ptr{Cvoid}},Ptr{Cvoid},Cstring) ref mod name
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

Base.unsafe_convert(::Type{Ptr{Cvoid}}, f::CuFunction) = f.ptr
