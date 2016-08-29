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
    "text": "Merlin is a deep learning framework written in Julia. It aims to provide a fast, flexible and compact deep learning library for machine learning.See README.md for basic usage.Basically,Wrap your data with Var type.\nApply functions to the Var.\nCompute gradient if necessary.x = Var(rand(Float32,10,5))\nzerograd!(x)\nf = Linear(Float32,10,3)\ny = f(x)\ngradient!(y)"
},

{
    "location": "functions.html#",
    "page": "Functions",
    "title": "Functions",
    "category": "page",
    "text": ""
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
    "location": "functions.html#Merlin.concat",
    "page": "Functions",
    "title": "Merlin.concat",
    "category": "Function",
    "text": "concat(dim::Int, xs::Var...)\nconcat(dim::Int, xs::Vector{Var})\n\nConcatenate arrays along the given dimension.\n\n\n\n"
},

{
    "location": "functions.html#Merlin.Conv",
    "page": "Functions",
    "title": "Merlin.Conv",
    "category": "Type",
    "text": "Conv(T, channel, filter, [stride, pad])\n\nN-dimensional convolution function.\n\nArguments\n\nT: Type\nfilterdims::NTuple{N,Int}: window size\nchanneldims::Tuple{Int,Int}: input channel, output channel\n[stride::NTuple{N,Int}]: stride size. Default: (1,1,...)\n[paddims::NTuple{N,Int}]: padding size. Default: (0,0,...)\n\n👉 Example\n\nx = Var(rand(Float32,5,4,3,2))\nf = Conv(Float32, (2,2), (3,4), stride=(1,1), paddims=(0,0))\ny = f(x)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.crossentropy",
    "page": "Functions",
    "title": "Merlin.crossentropy",
    "category": "Function",
    "text": "crossentropy(p,x)\n\nComputes cross-entropy between p and x. x is assumed to be unnormalized.\n\np: Vector{Int} or Matrix{Float}\n\n👉 Example\n\np = [1:5;]\nx = Var(rand(Float32,10,5))\ny = crossentropy(p,x)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.dropout",
    "page": "Functions",
    "title": "Merlin.dropout",
    "category": "Function",
    "text": "dropout(x::Var, ratio::Float64, istrain::Bool)\n\n\n\n"
},

{
    "location": "functions.html#Base.exp",
    "page": "Functions",
    "title": "Base.exp",
    "category": "Function",
    "text": "exp\n\n\n\n"
},

{
    "location": "functions.html#Merlin.gemm",
    "page": "Functions",
    "title": "Merlin.gemm",
    "category": "Function",
    "text": "gemm(tA::Char, tB::Char, alpha::Float64, A::Var, B::Var)\ngemm(A::Var, B::Var)\n\nC = alpha * tA(A) * tB(B)\n\nArguments\n\ntA: transpose ('T') or not ('N'). default: 'N'\ntB: the same as tA\n\n\n\n"
},

{
    "location": "functions.html#Base.getindex",
    "page": "Functions",
    "title": "Base.getindex",
    "category": "Function",
    "text": "getindex(x::Var, inds...)\n\n👉 Example\n\nx = Var(rand(Float32,10,5))\ny = x[1:3]\ny = x[2]\n\n\n\n"
},

{
    "location": "functions.html#Merlin.GRU",
    "page": "Functions",
    "title": "Merlin.GRU",
    "category": "Function",
    "text": "GRU(::Type, xsize::Int)\n\nGated Recurrent Unit (GRU). See: Chung et al. \"Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling\", 2014\n\nArguments\n\nxsize: size of input vector (= size of hidden vector)\n\n👉 Example\n\ngru = GRU(Float32,100)\nx = Var(rand(Float32,100))\nh = Var(rand(Float32,100))\ny = gru(x, h)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.Linear",
    "page": "Functions",
    "title": "Merlin.Linear",
    "category": "Type",
    "text": "Linear(w::Var, x::Var, [b::Var])\n\nCompute linear function (a.k.a. affine transformation).\n\nf(x) = W^Tx + b\n\nwhere W is a weight matrix and b is a bias vector.\n\n👉 Example\n\nx = Var(rand(Float32,10,5))\nf = Linear(Float32,10,7)\ny = f(x)\n\n\n\n"
},

{
    "location": "functions.html#Base.log",
    "page": "Functions",
    "title": "Base.log",
    "category": "Function",
    "text": "log\n\n\n\n"
},

{
    "location": "functions.html#Merlin.logsoftmax",
    "page": "Functions",
    "title": "Merlin.logsoftmax",
    "category": "Function",
    "text": "logsoftmax(x::Var, dim::Int)\n\n\n\n"
},

{
    "location": "functions.html#Base.max",
    "page": "Functions",
    "title": "Base.max",
    "category": "Function",
    "text": "max(x::Var, dim::Int)\n\nCompute the maximum value along the given dimensions.\n\n\n\n"
},

{
    "location": "functions.html#Merlin.maxpooling",
    "page": "Functions",
    "title": "Merlin.maxpooling",
    "category": "Function",
    "text": "maxpooling(window, [stride, padding])\n\nArguments\n\nwindims::NTuple{N,Int}: window size\nstride::NTuple{N,Int}: stride size. Default: (1,1,...)\npaddims::NTuple{N,Int}: padding size. Default: (0,0,...)\n\n👉 Example\n\nx = Var(rand(Float32,5,4,3,2))\ny = maxpooling(x, (3,3), stride=(1,1), paddims=(0,0))\n\n\n\n"
},

{
    "location": "functions.html#Merlin.relu",
    "page": "Functions",
    "title": "Merlin.relu",
    "category": "Function",
    "text": "relu(x::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.reshape",
    "page": "Functions",
    "title": "Base.reshape",
    "category": "Function",
    "text": "reshape(x::Var, dims::Int...)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.sigmoid",
    "page": "Functions",
    "title": "Merlin.sigmoid",
    "category": "Function",
    "text": "sigmoid(x::Var)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.softmax",
    "page": "Functions",
    "title": "Merlin.softmax",
    "category": "Function",
    "text": "softmax(x::Var, dim::Int)\n\n\n\n"
},

{
    "location": "functions.html#Base.sum",
    "page": "Functions",
    "title": "Base.sum",
    "category": "Function",
    "text": "sum(x, dim::Int)\n\nCompute the sum along the given dimensions.\n\n\n\n"
},

{
    "location": "functions.html#Base.tanh",
    "page": "Functions",
    "title": "Base.tanh",
    "category": "Function",
    "text": "tanh(x::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.transpose",
    "page": "Functions",
    "title": "Base.transpose",
    "category": "Function",
    "text": "transpose(x::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.view",
    "page": "Functions",
    "title": "Base.view",
    "category": "Function",
    "text": "view(x::Var, inds...)\n\n\n\n"
},

{
    "location": "functions.html#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": "+\n-\n*\n.*\naxsum\nconcat\nConv\ncrossentropy\ndropout\nexp\ngemm\ngetindex\nGRU\nLinear\nlog\nlogsoftmax\nmax\nmaxpooling\nrelu\nreshape\nsigmoid\nsoftmax\nsum\ntanh\ntranspose\nview"
},

{
    "location": "optimizers.html#",
    "page": "Optimizers",
    "title": "Optimizers",
    "category": "page",
    "text": ""
},

{
    "location": "optimizers.html#Merlin.AdaGrad",
    "page": "Optimizers",
    "title": "Merlin.AdaGrad",
    "category": "Type",
    "text": "AdaGrad\n\nSee: http://jmlr.org/papers/v12/duchi11a.html\n\n\n\n"
},

{
    "location": "optimizers.html#Merlin.Adam",
    "page": "Optimizers",
    "title": "Merlin.Adam",
    "category": "Type",
    "text": "Adam\n\nAdam: A Method for Stochastic Optimization See: http://arxiv.org/abs/1412.6980v8\n\n\n\n"
},

{
    "location": "optimizers.html#Merlin.SGD",
    "page": "Optimizers",
    "title": "Merlin.SGD",
    "category": "Type",
    "text": "SGD\n\nStochastic Gradient Descent.\n\nArguments\n\nrate: learning rate\nmomentum: momentum coefficient\n\n\n\n"
},

{
    "location": "optimizers.html#Optimizers-1",
    "page": "Optimizers",
    "title": "Optimizers",
    "category": "section",
    "text": "AdaGrad\nAdam\nSGD"
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
    "text": "Merlin supports saving and loading objects in HDF5 format."
},

{
    "location": "save_load.html#Merlin.h5save",
    "page": "Save and Load",
    "title": "Merlin.h5save",
    "category": "Function",
    "text": "h5save(filename::String, data)\n\nSave objects as a HDF5 format. Note that the objects are required to implement h5convert and h5load! functions.\n\n\n\n"
},

{
    "location": "save_load.html#Save-1",
    "page": "Save and Load",
    "title": "Save",
    "category": "section",
    "text": "h5saveFor example,x = Embeddings(Float32,10000,100)\nh5save(\"<filename>.h5\", x)A graph structure can be saved as well:T = Float32\nls = [Linear(T,10,7), Linear(T,7,3)]\ng = @graph begin\n    x = ls[1](:x)\n    x = relu(x)\n    x = ls[2](x)\n    x\nend\nh5save(\"<filename>.h5\", g)The saved HDF5 file is as follows: <p><img src=\"https://github.com/hshindo/Merlin.jl/blob/master/docs/src/assets/sample.h5.png\"></p>"
},

{
    "location": "save_load.html#Merlin.h5load",
    "page": "Save and Load",
    "title": "Merlin.h5load",
    "category": "Function",
    "text": "h5load(filename::String)\n\nLoad a HDF5 file.\n\n\n\n"
},

{
    "location": "save_load.html#Load-1",
    "page": "Save and Load",
    "title": "Load",
    "category": "section",
    "text": "h5load"
},

{
    "location": "save_load.html#Merlin.h5dict",
    "page": "Save and Load",
    "title": "Merlin.h5dict",
    "category": "Function",
    "text": "h5dict(T::Type, x::Pair...)\n\nCreate a hdf5 dictionary with type information.\n\n\n\n"
},

{
    "location": "save_load.html#Saving-Your-Own-Objects-1",
    "page": "Save and Load",
    "title": "Saving Your Own Objects",
    "category": "section",
    "text": "It requires to implement h5convert and h5load! functions.h5dict"
},

]}
