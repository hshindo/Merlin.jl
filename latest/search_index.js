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
    "text": "Merlin is a flexible deep learning framework written in Julia. It aims to provide a fast, flexible and compact deep learning library for machine learning."
},

{
    "location": "index.html#Requirements-1",
    "page": "Home",
    "title": "Requirements",
    "category": "section",
    "text": "Julia 0.6\ng++ (for OSX or Linux)"
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "julia> Pkg.add(\"Merlin\")"
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
    "text": "Pages = [\"functions.md\"]"
},

{
    "location": "functions.html#Merlin.crelu",
    "page": "Functions",
    "title": "Merlin.crelu",
    "category": "Function",
    "text": "crelu(x)\n\nConcatenated Rectified Linear Unit. The output is twice the size of the input.\n\nf(x) = (max(0x) max(0-x))\n\n\n\n"
},

{
    "location": "functions.html#Merlin.elu",
    "page": "Functions",
    "title": "Merlin.elu",
    "category": "Function",
    "text": "elu(x)\n\nExponential Linear Unit.\n\nf(x) =\nbegincases\nx  x  0 \nalpha (e^x-1)  xleq0\nendcases\n\n\n\n"
},

{
    "location": "functions.html#Merlin.relu",
    "page": "Functions",
    "title": "Merlin.relu",
    "category": "Function",
    "text": "relu(x)\n\nRectified Linear Unit.\n\nf(x) = max(0 x)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.selu",
    "page": "Functions",
    "title": "Merlin.selu",
    "category": "Function",
    "text": "selu(x)\n\nScaled Exponential Linear Unit.\n\nf(x) = lambda\nbegincases\nx  x  0 \nalpha e^x-alpha  xleq0\nendcases\n\nReferences\n\nKlambauer et al., Self-Normalizing Neural Networks, NIPS 2017\n\n\n\n"
},

{
    "location": "functions.html#Merlin.sigmoid",
    "page": "Functions",
    "title": "Merlin.sigmoid",
    "category": "Function",
    "text": "sigmoid(x)\n\nSigmoid logistic function.\n\nf(x) = (1 + exp(-x))^-1\n\n\n\n"
},

{
    "location": "functions.html#Base.tanh",
    "page": "Functions",
    "title": "Base.tanh",
    "category": "Function",
    "text": "tanh(x)\n\nHyperbolic tangent function.\n\n\n\n"
},

{
    "location": "functions.html#Activation-1",
    "page": "Functions",
    "title": "Activation",
    "category": "section",
    "text": "crelu\nelu\nrelu\nselu\nsigmoid\ntanh"
},

{
    "location": "functions.html#Merlin.Conv1D",
    "page": "Functions",
    "title": "Merlin.Conv1D",
    "category": "Type",
    "text": "Conv1D(T, ksize, insize, outsize, pad, stride; dilation=1, [init_w=Xavier()], [init_b=Zeros()])\n\n1-dimensional convolution function.\n\nx = Var(rand(Float32,10,5))\nf = Conv1D(Float32, 5, 10, 3, 2, 1)\ny = f(x)\n\n\n\n"
},

{
    "location": "functions.html#Convolution-1",
    "page": "Functions",
    "title": "Convolution",
    "category": "section",
    "text": "Conv1D"
},

{
    "location": "functions.html#Merlin.crossentropy",
    "page": "Functions",
    "title": "Merlin.crossentropy",
    "category": "Function",
    "text": "crossentropy(p, q)\n\nCross-entropy function between p and q.\n\nf(x) = -sum_x p(x) log q(x)\n\np::Var: Var of Vector{Int} or Matrix{Float}. If p is Vector{Int} and p[i] == 0, returns 0.\nq::Var: Var of Matrix{Float}\n\np = Var(rand(0:10,5))\nq = softmax(Var(rand(Float32,10,5)))\ny = crossentropy(p, q)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.mse",
    "page": "Functions",
    "title": "Merlin.mse",
    "category": "Function",
    "text": "mse(x1, x2)\n\nMean Squared Error function between x1 and x2. The mean is calculated over the minibatch. Note that the error is not scaled by 1/2.\n\n\n\n"
},

{
    "location": "functions.html#Merlin.softmax_crossentropy",
    "page": "Functions",
    "title": "Merlin.softmax_crossentropy",
    "category": "Function",
    "text": "softmax_crossentropy(p, x)\n\nCross-entropy function between p and softmax(x).\n\nf(x) = -sum_x p(x) log q(x)\n\nwhere q = softmax(x)\n\np: Var of Vector{Int} or Matrix{Float}\nq: Var of Matrix{Float}\n\np = Var(rand(0:10,5))\nq = Var(rand(Float32,10,5))\ny = softmax_crossentropy(p, x)\n\n\n\n"
},

{
    "location": "functions.html#Loss-1",
    "page": "Functions",
    "title": "Loss",
    "category": "section",
    "text": "crossentropy\nmse\nsoftmax_crossentropy"
},

{
    "location": "functions.html#Base.broadcast",
    "page": "Functions",
    "title": "Base.broadcast",
    "category": "Function",
    "text": ".+(x1::Var, x2::Var)\n\n\n\n.-(x1::Var, x2::Var)\n\n\n\n.*(x1::Var, x2::Var)\n\n\n\n"
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
    "text": "-(x1, x2)\n\n\n\n"
},

{
    "location": "functions.html#Base.:*",
    "page": "Functions",
    "title": "Base.:*",
    "category": "Function",
    "text": "*(A::Var, B::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.:/",
    "page": "Functions",
    "title": "Base.:/",
    "category": "Function",
    "text": "/(x1::Var, a)\n\n\n\n"
},

{
    "location": "functions.html#Base.:^",
    "page": "Functions",
    "title": "Base.:^",
    "category": "Function",
    "text": "^(x::Var, a::Number)\n\n\n\n"
},

{
    "location": "functions.html#Base.transpose",
    "page": "Functions",
    "title": "Base.transpose",
    "category": "Function",
    "text": "transpose(x)\n\n\n\n"
},

{
    "location": "functions.html#Math-1",
    "page": "Functions",
    "title": "Math",
    "category": "section",
    "text": "broadcast\n+\n-\n*\n/\n^\ntranspose"
},

{
    "location": "functions.html#Merlin.dropout",
    "page": "Functions",
    "title": "Merlin.dropout",
    "category": "Function",
    "text": "dropout(x::Var, rate::Float64)\n\nIf config.train is true, drops elements randomly with probability rate and scales the other elements by factor 1  (1 - rate). Otherwise, it just returns x.\n\n\n\n"
},

{
    "location": "functions.html#Random-1",
    "page": "Functions",
    "title": "Random",
    "category": "section",
    "text": "dropout"
},

{
    "location": "functions.html#Merlin.BiLSTM",
    "page": "Functions",
    "title": "Merlin.BiLSTM",
    "category": "Type",
    "text": "BiLSTM(x)\n\nBidirectional Long Short-Term Memory network.\n\n👉 Example\n\nx = Var(rand(Float32,10,5))\n\n\n\n"
},

{
    "location": "functions.html#Merlin.LSTM",
    "page": "Functions",
    "title": "Merlin.LSTM",
    "category": "Type",
    "text": "LSTM(T::Type, xsize::Int, hsize::Int)\n\nLong Short-Term Memory network.\n\nT = Float32\nlstm = LSTM(T, 100, 100)\nh = lstm(x)\n\n\n\n"
},

{
    "location": "functions.html#Recurrent-1",
    "page": "Functions",
    "title": "Recurrent",
    "category": "section",
    "text": "BiLSTM\nLSTM"
},

{
    "location": "functions.html#Reduction-1",
    "page": "Functions",
    "title": "Reduction",
    "category": "section",
    "text": "max\nmax_batchgemm\ngemv\nconcat\ngetindex\nLinear\nlogsoftmax\nlookup\nreshape\nsoftmax\nstandardize\nwindow1d"
},

{
    "location": "initializers.html#",
    "page": "Initializaters",
    "title": "Initializaters",
    "category": "page",
    "text": ""
},

{
    "location": "initializers.html#Merlin.Uniform",
    "page": "Initializaters",
    "title": "Merlin.Uniform",
    "category": "Type",
    "text": "Uniform(a, b)\n\nGenerator of ndarray with a uniform distribution.\n\nArguments\n\na: Lower bound of the range of random values.\nb: Upper bound of the range of random values.\n\n\n\n"
},

{
    "location": "initializers.html#Merlin.Normal",
    "page": "Initializaters",
    "title": "Merlin.Normal",
    "category": "Type",
    "text": "Normal(mean, var)\n\nGenerator of ndarray with a normal distribution.\n\nArguments\n\nmean: Mean of the random values.\nvar: Variance of the random values.\n\n\n\n"
},

{
    "location": "initializers.html#Merlin.Xavier",
    "page": "Initializaters",
    "title": "Merlin.Xavier",
    "category": "Type",
    "text": "Xavier()\n\nXavier initialization.\n\n\n\n"
},

{
    "location": "initializers.html#Merlin.Orthogonal",
    "page": "Initializaters",
    "title": "Merlin.Orthogonal",
    "category": "Type",
    "text": "Orthogonal()\n\nReferences\n\nSaxe et al., Exact solutions to the nonlinear dynamics of learning in deep linear neural networks\n\n\n\n"
},

{
    "location": "initializers.html#Initializers-1",
    "page": "Initializaters",
    "title": "Initializers",
    "category": "section",
    "text": "Pages = [\"initializers.md\"]Uniform\nNormal\nXavier\nOrthogonal"
},

{
    "location": "initializers.html#Custom-Initializer-1",
    "page": "Initializaters",
    "title": "Custom Initializer",
    "category": "section",
    "text": "import Merlin.random\n\nstruct CustomRand\nend\n\nfunction random{T}(init, ::Type{T}, dims...)\n    # code\nend"
},

]}
