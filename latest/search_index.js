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
    "text": "Merlin is a flexible deep learning framework written in Julia. It aims to provide a fast, flexible and compact deep learning library for machine learning.See README.md for basic usage."
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
    "text": "+\n-\n*\n.*\nconcat\nConv\ncrossentropy\ndropout\nexp\ngemm\ngetindex\nGRU\nLinear\nlog\nlogsoftmax\nmax\nmaxpooling\nrelu\nreshape\nsigmoid\nsoftmax\nsum\ntanh\ntranspose\nview"
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
    "text": "SGD\n\nStochastic Gradient Descent.\n\nArguments\n\nrate: learning rate\n[momentum]: momentum coefficient\n\n\n\n"
},

{
    "location": "optimizers.html#Optimizers-1",
    "page": "Optimizers",
    "title": "Optimizers",
    "category": "section",
    "text": "A Optimizer provides functions for updating parameters.For example,x1 = Var(rand(Float32,5,4))\nx1.grad = rand(Float32,5,4)\nopt = SGD(0.001)\nopt(x1.data, x1.grad)AdaGrad\nAdam\nSGD"
},

{
    "location": "save_load.html#",
    "page": "Save and Load",
    "title": "Save and Load",
    "category": "page",
    "text": ""
},

{
    "location": "save_load.html#Merlin.save",
    "page": "Save and Load",
    "title": "Merlin.save",
    "category": "Function",
    "text": "save(path::String, mode::String, name::String, obj)\n\nSave an object in Merlin HDF5 format.\n\nmode: \"w\" (overrite) or \"r+\" (append)\n\n\n\n"
},

{
    "location": "save_load.html#Merlin.load",
    "page": "Save and Load",
    "title": "Merlin.load",
    "category": "Function",
    "text": "load(path::String, name::String)\n\nLoad an object from Merlin HDF5 format.\n\n\n\n"
},

{
    "location": "save_load.html#Save-and-Load-1",
    "page": "Save and Load",
    "title": "Save and Load",
    "category": "section",
    "text": "Merlin supports saving and loading objects in HDF5 format.For saving objects provided by Merlin, use Merlin.save and Merlin.load functions.\nFor other complex objects, it is recommended to use JLD.save and JLD.load functions provided by JLD.jl.Merlin.save\nMerlin.loadFor example,x = Embeddings(Float32,10000,100)\nMerlin.save(\"embedding.h5\", \"w\", \"x\", x)A graph structure can be saved as well:T = Float32\nx = Var()\ny = Linear(T,10,7)(x)\ny = relu(y)\ny = Linear(T,7,3)(y)\ng = Graph(y, x)\nMerlin.save(\"graph.h5\", \"g\", g)The saved HDF5 file is as follows:(Image: graph.h5)"
},

{
    "location": "save_load.html#Custom-Serialization-1",
    "page": "Save and Load",
    "title": "Custom Serialization",
    "category": "section",
    "text": "It requires to implement h5convert function for custom serialization/deserialization. See Merlin sources for details."
},

]}
