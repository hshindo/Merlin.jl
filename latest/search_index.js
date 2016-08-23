var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Merlin.jl-1",
    "page": "Home",
    "title": "Merlin.jl",
    "category": "section",
    "text": "Merlin is a deep learning framework written in Julia. It aims to provide a fast, flexible and compact deep learning library for machine learning.See README.md for basic usage.Basically,Wrap your data with Var type.\nApply functions to the Var.x = Var(rand(Float32,10,5))\nf = Linear(Float32,10,3)\ny = f(x)"
},

{
    "location": "functions.html#",
    "page": "Functions",
    "title": "Functions",
    "category": "page",
    "text": ""
},

{
    "location": "functions.html#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": "Merlin provides standard functions used in deep learning."
},

{
    "location": "functions.html#Index-1",
    "page": "Functions",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"functions.md\"]"
},

{
    "location": "functions.html#Merlin.relu",
    "page": "Functions",
    "title": "Merlin.relu",
    "category": "Function",
    "text": "relu(x::Var)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.sigmoid",
    "page": "Functions",
    "title": "Merlin.sigmoid",
    "category": "Function",
    "text": "sigmoid(x::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.tanh",
    "page": "Functions",
    "title": "Base.tanh",
    "category": "Function",
    "text": "tanh(x::Var)\n\n\n\n"
},

{
    "location": "functions.html#Activation-Functions-1",
    "page": "Functions",
    "title": "Activation Functions",
    "category": "section",
    "text": "relu\nsigmoid\ntanh"
},

{
    "location": "functions.html#Base.getindex",
    "page": "Functions",
    "title": "Base.getindex",
    "category": "Function",
    "text": "getindex(x::Var, inds...)\n\n👉 Example\n\nx = Var(rand(Float32,10,5))\ny = x[1:3]\ny = x[2]\n\n\n\n"
},

{
    "location": "functions.html#Base.view",
    "page": "Functions",
    "title": "Base.view",
    "category": "Function",
    "text": "view(x::Var, inds...)\n\n\n\n"
},

{
    "location": "functions.html#Indexing-Functions-1",
    "page": "Functions",
    "title": "Indexing Functions",
    "category": "section",
    "text": "getindex\nview"
},

{
    "location": "functions.html#Merlin.concat",
    "page": "Functions",
    "title": "Merlin.concat",
    "category": "Function",
    "text": "concat(dim::Int, xs::Var...)\nconcat(dim::Int, xs::Vector{Var})\n\nConcatenate arrays along the given dimension.\n\n\n\n"
},

{
    "location": "functions.html#Base.reshape",
    "page": "Functions",
    "title": "Base.reshape",
    "category": "Function",
    "text": "reshape(x::Var, dims::Int...)\n\n\n\n"
},

{
    "location": "functions.html#Base.transpose",
    "page": "Functions",
    "title": "Base.transpose",
    "category": "Function",
    "text": "transpose(x::Var)\n\n\n\n"
},

{
    "location": "functions.html#Manipulation-Functions-1",
    "page": "Functions",
    "title": "Manipulation Functions",
    "category": "section",
    "text": "concat\nreshape\ntranspose"
},

{
    "location": "functions.html#Base.:+",
    "page": "Functions",
    "title": "Base.:+",
    "category": "Function",
    "text": "+(x1::Var, x2::Var)\n+(a::Number, x::Var)\n+(x::Var, a::Number)\n\n\n\n"
},

{
    "location": "functions.html#Base.:-",
    "page": "Functions",
    "title": "Base.:-",
    "category": "Function",
    "text": "-(x1::Var, x2::Var)\n-(a::Number, x::Var)\n-(a::Number, x::Var)\n-(x::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.:*",
    "page": "Functions",
    "title": "Base.:*",
    "category": "Function",
    "text": "*(x1::Var, x2::Var)\n*(a::Number, x::Var)\n*(x::Var, a::Number)\n\n\n\n"
},

{
    "location": "functions.html#Base.:.*",
    "page": "Functions",
    "title": "Base.:.*",
    "category": "Function",
    "text": ".*(x1::Var, x2::Var)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.axsum",
    "page": "Functions",
    "title": "Merlin.axsum",
    "category": "Function",
    "text": "axsum\n\ny=sum_ialeftiightcdot xleftiight\n\nwhere ai is a scalar and x is scholar or vector. Every operation is broadcasted.\n\n\n\n"
},

{
    "location": "functions.html#Base.exp",
    "page": "Functions",
    "title": "Base.exp",
    "category": "Function",
    "text": "exp\n\n\n\n"
},

{
    "location": "functions.html#Base.log",
    "page": "Functions",
    "title": "Base.log",
    "category": "Function",
    "text": "log\n\n\n\n"
},

{
    "location": "functions.html#Base.max",
    "page": "Functions",
    "title": "Base.max",
    "category": "Function",
    "text": "max(x::Var, dim::Int)\n\nCompute the maximum value along the given dimensions.\n\n\n\n"
},

{
    "location": "functions.html#Base.sum",
    "page": "Functions",
    "title": "Base.sum",
    "category": "Function",
    "text": "sum(x, dim::Int)\n\nCompute the sum along the given dimensions.\n\n\n\n"
},

{
    "location": "functions.html#Math-Functions-1",
    "page": "Functions",
    "title": "Math Functions",
    "category": "section",
    "text": "+\n-\n*\n.*\naxsum\nexp\nlog\nmax\nsum"
},

{
    "location": "functions.html#Merlin.maxpooling",
    "page": "Functions",
    "title": "Merlin.maxpooling",
    "category": "Function",
    "text": "maxpooling(window, [stride, padding])\n\nArguments\n\nwindims::NTuple{N,Int}: window size\nstride::NTuple{N,Int}: stride size. Default: (1,1,...)\npaddims::NTuple{N,Int}: padding size. Default: (0,0,...)\n\n👉 Example\n\nx = Var(rand(Float32,5,4,3,2))\ny = maxpooling(x, (3,3), stride=(1,1), paddims=(0,0))\n\n\n\n"
},

{
    "location": "functions.html#Pooling-Functions-1",
    "page": "Functions",
    "title": "Pooling Functions",
    "category": "section",
    "text": "maxpooling"
},

{
    "location": "functions.html#Merlin.logsoftmax",
    "page": "Functions",
    "title": "Merlin.logsoftmax",
    "category": "Function",
    "text": "logsoftmax(x::Var, dim::Int)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.softmax",
    "page": "Functions",
    "title": "Merlin.softmax",
    "category": "Function",
    "text": "softmax(x::Var, dim::Int)\n\n\n\n"
},

{
    "location": "functions.html#Softmax-Functions-1",
    "page": "Functions",
    "title": "Softmax Functions",
    "category": "section",
    "text": "logsoftmax\nsoftmax"
},

{
    "location": "save_load.html#",
    "page": "Save and Load",
    "title": "Save and Load",
    "category": "page",
    "text": ""
},

{
    "location": "save_load.html#Save-and-Load-1",
    "page": "Save and Load",
    "title": "Save and Load",
    "category": "section",
    "text": "Merlin supports saving and loading objects (network structures, parameters, etc.) in HDF5 format."
},

{
    "location": "save_load.html#Save-1",
    "page": "Save and Load",
    "title": "Save",
    "category": "section",
    "text": "x =\nMerlin.save(\"<filename>\", x)"
},

{
    "location": "save_load.html#Load-1",
    "page": "Save and Load",
    "title": "Load",
    "category": "section",
    "text": "To deserialize objects,Merlin.load()"
},

{
    "location": "save_load.html#Index-1",
    "page": "Save and Load",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"functions.md\"]"
},

{
    "location": "save_load.html#Activation-Functions-1",
    "page": "Save and Load",
    "title": "Activation Functions",
    "category": "section",
    "text": "relu\nsigmoid\ntanh"
},

{
    "location": "save_load.html#How-to-serialize-your-object?-1",
    "page": "Save and Load",
    "title": "How to serialize your object?",
    "category": "section",
    "text": "It requires * HDF5Dict(x) * load_hdf5(::Type{T}, x::Dict)See examples."
},

]}
