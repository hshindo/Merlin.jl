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
    "location": "var.html#",
    "page": "Var",
    "title": "Var",
    "category": "page",
    "text": ""
},

{
    "location": "var.html#Merlin.Var",
    "page": "Var",
    "title": "Merlin.Var",
    "category": "Type",
    "text": "Var\n\nVariable struct.\n\nVar contains the following members:\n\ndata\nbatchdims\nf\nargs\ngrad\n\n\n\n"
},

{
    "location": "var.html#Merlin.batch-Tuple{Array{Merlin.Var,1},Int64}",
    "page": "Var",
    "title": "Merlin.batch",
    "category": "Method",
    "text": "batch(data::Vector{Var}, batchsize::Int)\nbatch(data::Vector{NTuple{N,Var}}, batchsize::Int) where N\n\nCreate batch from variables.\n\n\n\n"
},

{
    "location": "var.html#Merlin.batchsize-Tuple{Merlin.Var}",
    "page": "Var",
    "title": "Merlin.batchsize",
    "category": "Method",
    "text": "batchsize(x::Var)\nbatchsize(x::Var, i::Int)\nbatchsize(x::Node)\nbatchsize(x::Node, i::Int)\n\n\n\n"
},

{
    "location": "var.html#Merlin.gradient!-Tuple{Merlin.Var}",
    "page": "Var",
    "title": "Merlin.gradient!",
    "category": "Method",
    "text": "gradient!(top::Var)\n\nCompute gradients.\n\n\n\n"
},

{
    "location": "var.html#Merlin.isparam-Tuple{Merlin.Var}",
    "page": "Var",
    "title": "Merlin.isparam",
    "category": "Method",
    "text": "isparam(x::Var)::Bool\n\nReturns whether x is a parameter or not\n\n\n\n"
},

{
    "location": "var.html#Merlin.topsort-Union{Tuple{T}, Tuple{Vararg{T,N} where N}} where T",
    "page": "Var",
    "title": "Merlin.topsort",
    "category": "Method",
    "text": "topsort\n\nTopological sort.\n\n\n\n"
},

{
    "location": "var.html#Var-1",
    "page": "Var",
    "title": "Var",
    "category": "section",
    "text": "Modules = [Merlin]\nPages = [\"var.jl\"]"
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
    "text": "LSTM(::Type{T}, insize::Int, outsize::Int, [init_W=Uniform(0.001), init_U=Orthogonal()])\n\nLong Short-Term Memory network.\n\nT = Float32\nf = LSTM(T, 100, 100)\nh = f(x)\n\n\n\n"
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
    "location": "graph.html#",
    "page": "Graph",
    "title": "Graph",
    "category": "page",
    "text": ""
},

{
    "location": "graph.html#Graph-1",
    "page": "Graph",
    "title": "Graph",
    "category": "section",
    "text": "Graph represents a computational graph.using Merlin\n\nT = Float32\nx = Node()\ny = Linear(T,10,7)(x)\ny = relu(y)\ny = Linear(T,7,3)(y)\n@assert typeof(y) == Node\ng = Graph(input=x, output=y)\n\nx = zerograd(rand(T,10,10))\ny = g(x)\n\nparams = gradient!(y)\nprintln(x.grad)\n\nopt = SGD(0.01)\nforeach(opt, params)"
},

{
    "location": "initializers.html#",
    "page": "Initializaters",
    "title": "Initializaters",
    "category": "page",
    "text": ""
},

{
    "location": "initializers.html#Initializers-1",
    "page": "Initializaters",
    "title": "Initializers",
    "category": "section",
    "text": "Modules = [Merlin]\nPages = [\"initializers.md\"]Modules = [Merlin]\nPages = [\"initializers.jl\"]"
},

{
    "location": "optimizers.html#",
    "page": "Optimizers",
    "title": "Optimizers",
    "category": "page",
    "text": ""
},

{
    "location": "optimizers.html#Optimizers-1",
    "page": "Optimizers",
    "title": "Optimizers",
    "category": "section",
    "text": "x = zerograd(rand(Float32,5,4))\nopt = SGD(0.001)\nopt(x)Modules = [Merlin]\nPages = [\"optimizers.md\"]Modules = [Merlin]\nPages = [\"optimizers.jl\"]"
},

{
    "location": "datasets.html#",
    "page": "Datasets",
    "title": "Datasets",
    "category": "page",
    "text": ""
},

{
    "location": "datasets.html#Datasets-1",
    "page": "Datasets",
    "title": "Datasets",
    "category": "section",
    "text": "Merlin provides an access to common machine learning datasets for Julia."
},

{
    "location": "datasets.html#Example-1",
    "page": "Datasets",
    "title": "Example",
    "category": "section",
    "text": "using Merlin\nusing Merlin.Datasets\nusing Merlin.Datasets.MNIST\n\ndir = \"mnist\"\ntrain_x, train_y = MNIST.traindata(dir)\ntest_x, test_y = MNIST.testdata(dir)"
},

{
    "location": "datasets.html#Available-Datasets-1",
    "page": "Datasets",
    "title": "Available Datasets",
    "category": "section",
    "text": ""
},

{
    "location": "datasets.html#CIFAR10-1",
    "page": "Datasets",
    "title": "CIFAR10",
    "category": "section",
    "text": "The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes."
},

{
    "location": "datasets.html#CIFAR100-1",
    "page": "Datasets",
    "title": "CIFAR100",
    "category": "section",
    "text": "The CIFAR-100 dataset consists of 600 32x32 color images in 100 classes. The 100 classes are grouped into 20 superclasses (fine and coarse labels)."
},

{
    "location": "datasets.html#MNIST-1",
    "page": "Datasets",
    "title": "MNIST",
    "category": "section",
    "text": "The MNIST dataset consists of 60000 28x28 images of handwritten digits."
},

{
    "location": "datasets.html#PTBLM-1",
    "page": "Datasets",
    "title": "PTBLM",
    "category": "section",
    "text": "The PTBLM dataset consists of Penn Treebank sentences for language modeling, available from tomsercu/lstm. The unknown words are replaced with <unk> so that the total vocaburary size becomes 10000.This is the first sentence of the PTBLM dataset.dir = \"ptblm\"\nx, y = PTBLM.traindata(dir)\n\nx[1]\n> [\"no\", \"it\", \"was\", \"n't\", \"black\", \"monday\"]\ny[1]\n> [\"it\", \"was\", \"n't\", \"black\", \"monday\", \"<eos>\"]where MLDataset adds the special word: <eos> to the end of y."
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
    "text": "It is recommended to use JLD2 for object serialization."
},

]}
