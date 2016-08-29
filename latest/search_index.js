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
    "location": "functions.html#Functions-1",
    "page": "Functions",
    "title": "Functions",
    "category": "section",
    "text": "\"add\"\n\"axsum\"\n\"concat\"\n\"conv\"\n\"crossentropy\"\n\"dropout\"\n\"embedding\"\n\"exp\"\n\"gemm\"\n\"getindex\"\n\"gru\"\n\"linear\"\n\"log\"\n\"logsoftmax\"\n\"max\"\n\"maxpooling\"\n\"multiply\"\n\"norm\"\n\"relu\"\n\"reshape\"\n\"sigmoid\"\n\"softmax\"\n\"sum\"\n\"tanh\"\n\"transpose\"\n\"view\"\n\"window2\""
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
