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
    "category": "type",
    "text": "Var\n\nVariable struct.\n\nVar contains the following members:\n\ndata\nargs\ngrad\nwork\n\nExample\n\nT = Float32\nx = Var(rand(T,10,5)) # x.grad is set to `nothing`\nx = zerograd(rand(T,10,5)) # x.grad is initialized as zero.\n\n\n\n"
},

{
    "location": "var.html#Merlin.gradient!-Tuple{Vararg{Merlin.Var,N} where N}",
    "page": "Var",
    "title": "Merlin.gradient!",
    "category": "method",
    "text": "gradient!(top::Var)\n\nCompute gradients.\n\n\n\n"
},

{
    "location": "var.html#Merlin.isparam-Tuple{Merlin.Var}",
    "page": "Var",
    "title": "Merlin.isparam",
    "category": "method",
    "text": "isparam(x::Var)\n\nReturns whether x is a parameter or not\n\n\n\n"
},

{
    "location": "var.html#Merlin.topsort-Union{Tuple{T}, Tuple{Vararg{T,N} where N}} where T",
    "page": "Var",
    "title": "Merlin.topsort",
    "category": "method",
    "text": "topsort(tops::T...)\n\nTopological sort.\n\n\n\n"
},

{
    "location": "var.html#Var-1",
    "page": "Var",
    "title": "Var",
    "category": "section",
    "text": "Var is a type of variable for keeping gradients and a history of function calls.Modules = [Merlin]\nPages = [\"var.jl\"]"
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
    "location": "functions.html#Merlin.crelu-Tuple{Merlin.Var}",
    "page": "Functions",
    "title": "Merlin.crelu",
    "category": "method",
    "text": "crelu(x::Var)\n\nConcatenated Rectified Linear Unit. The output is twice the size of the input.\n\nf(x) = (max(0x) max(0-x))\n\nReferences\n\nShang et al., \"Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units\", arXiv 2016.\n\n\n\n"
},

{
    "location": "functions.html#Merlin.elu-Tuple{Merlin.Var}",
    "page": "Functions",
    "title": "Merlin.elu",
    "category": "method",
    "text": "elu(x::Var)\n\nExponential Linear Unit.\n\nReferences\n\nClevert et al., \"Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)\", arXiv 2015.\n\nf(x) =\nbegincases\nx  x  0 \nalpha (e^x-1)  xleq0\nendcases\n\nwhere alpha=1.\n\n\n\n"
},

{
    "location": "functions.html#Merlin.leaky_relu",
    "page": "Functions",
    "title": "Merlin.leaky_relu",
    "category": "function",
    "text": "leaky_relu(x::Var, alpha=0.1)\n\nLeaky Rectified Linear Unit.\n\nf(x) =\nbegincases\nx  x  0 \nalpha x  x leq 0\nendcases\n\nReferences\n\nMaas et al., \"Rectifier Nonlinearities Improve Neural Network Acoustic Models\", ICML 2013.\n\n\n\n"
},

{
    "location": "functions.html#Merlin.relu-Tuple{Merlin.Var}",
    "page": "Functions",
    "title": "Merlin.relu",
    "category": "method",
    "text": "relu(x::Var)\n\nRectified Linear Unit.\n\nf(x) = max(0 x)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.selu-Tuple{Merlin.Var}",
    "page": "Functions",
    "title": "Merlin.selu",
    "category": "method",
    "text": "selu(x::Var)\n\nScaled Exponential Linear Unit.\n\nf(x) = lambda\nbegincases\nx  x  0 \nalpha e^x-alpha  xleq0\nendcases\n\nwhere lambda=10507 and alpha=16733.\n\nReferences\n\nKlambauer et al., \"Self-Normalizing Neural Networks\", NIPS 2017.\n\n\n\n"
},

{
    "location": "functions.html#Merlin.sigmoid-Tuple{Merlin.Var}",
    "page": "Functions",
    "title": "Merlin.sigmoid",
    "category": "method",
    "text": "sigmoid(x)\n\nSigmoid logistic function.\n\nf(x) = (1 + exp(-x))^-1\n\n\n\n"
},

{
    "location": "functions.html#Merlin.Swish",
    "page": "Functions",
    "title": "Merlin.Swish",
    "category": "type",
    "text": "Swish\n\nSwish activation function.\n\nf(x) = x cdot sigma (beta x)\n\nwhere beta is a leanable parameter.\n\nReferences\n\nRamachandran et al. \"Searching for Activation Functions\", arXiv 2017.\n\n\n\n"
},

{
    "location": "functions.html#Base.tanh-Tuple{Merlin.Var}",
    "page": "Functions",
    "title": "Base.tanh",
    "category": "method",
    "text": "tanh(x::Var)\n\nHyperbolic tangent function.\n\n\n\n"
},

{
    "location": "functions.html#Activation-1",
    "page": "Functions",
    "title": "Activation",
    "category": "section",
    "text": "Modules = [Merlin]\nPages   = [\"activation.jl\"]"
},

{
    "location": "functions.html#Merlin.gemm_batch-Tuple{Char,Char,Any,Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Merlin.gemm_batch",
    "category": "method",
    "text": "gemm_batch(tA::Char, tB::Char, alpha, A::Var, B::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.LinAlg.BLAS.gemm-Tuple{Char,Char,Number,Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Base.LinAlg.BLAS.gemm",
    "category": "method",
    "text": "gemm(tA::Char, tB::Char, alpha, A::Var, B::Var)\ngemm(A::Var, B::Var, [tA=\'N\'], [tB=\'N\'], [alpha=1])\n\ntA, tB: \'T\' (transpose) or \'N\' (not transpose)\n\nC = alpha times textrmtA(A) times textrmtB(B)\n\nT = Float32\nA = Var(rand(T,10,5))\nB = Var(rand(T,10,7))\nC = BLAS.gemm(\'T\', \'N\', 1, A, B)\n\n\n\n"
},

{
    "location": "functions.html#Base.LinAlg.BLAS.gemv-Tuple{Char,Number,Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Base.LinAlg.BLAS.gemv",
    "category": "method",
    "text": "BLAS.gemv(tA::Char, alpha, A::Var, x::Var)\n\ntA: \'T\' (transpose) or \'N\' (not transpose)\n\ny = alpha times textrmtA(A) times x\n\nT = Float32\nA = Var(rand(T,10,5))\nx = Var(rand(T,5))\nB = BLAS.gemv(\'N\', 1, A, x)\n\n\n\n"
},

{
    "location": "functions.html#BLAS-1",
    "page": "Functions",
    "title": "BLAS",
    "category": "section",
    "text": "Modules = [Merlin]\nPages   = [\"blas.jl\"]"
},

{
    "location": "functions.html#Merlin.Conv",
    "page": "Functions",
    "title": "Merlin.Conv",
    "category": "type",
    "text": "Conv(T, filtersize, kwargs...)\n\nW: (W1,W2,...,I,O)\nX: (X1,X2,...,I,N)\nY: (Y1,Y2,...,O,N)\n\nwhere\n\nI: number of input channels\nO: number of output channels\nN: batch size\n\nT = Float32\nconv = Conv(T, (1,1,3,2))\nx = CuArray{T}(5,5,3,3)\ny = conv(x)\n\n\n\n"
},

{
    "location": "functions.html#Convolution-1",
    "page": "Functions",
    "title": "Convolution",
    "category": "section",
    "text": "Modules = [Merlin]\nPages   = [\"conv.jl\"]"
},

{
    "location": "functions.html#Embeddings-1",
    "page": "Functions",
    "title": "Embeddings",
    "category": "section",
    "text": "Modules = [Merlin]\nPages   = [\"embeddings.jl\"]"
},

{
    "location": "functions.html#Merlin.crossentropy-Tuple{Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Merlin.crossentropy",
    "category": "method",
    "text": "crossentropy(p, q)\n\nCross-entropy function between p and q.\n\nf(x) = -sum_x p(x) log q(x)\n\np::Var: Var of Vector{Int} or Matrix{Float}. If p is Vector{Int} and p[i] == 0, returns 0.\nq::Var: Var of Matrix{Float}\n\np = Var(rand(0:10,5))\nq = softmax(Var(rand(Float32,10,5)))\ny = crossentropy(p, q)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.l2-Tuple{Merlin.Var,Float64}",
    "page": "Functions",
    "title": "Merlin.l2",
    "category": "method",
    "text": "l2(x::Var, lambda::Float64)\n\nL2 regularization.\n\ny = fraclambda2leftVert mathbfx rightVert ^2\n\nx = Var(rand(Float32,10,5))\ny = l2(x, 0.01)\n\n\n\n"
},

{
    "location": "functions.html#Merlin.mse-Tuple{Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Merlin.mse",
    "category": "method",
    "text": "mse(x1, x2)\n\nMean Squared Error function between x1 and x2. The mean is calculated over the minibatch. Note that the error is not scaled by 1/2.\n\n\n\n"
},

{
    "location": "functions.html#Merlin.softmax_crossentropy-Tuple{Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Merlin.softmax_crossentropy",
    "category": "method",
    "text": "softmax_crossentropy(p, x)\n\nCross-entropy function between p and softmax(x).\n\nf(x) = -sum_x p(x) log q(x)\n\nwhere q = softmax(x)\n\np: Var of Vector{Int} or Matrix{Float}\nq: Var of Matrix{Float}\n\np = Var(rand(0:10,5))\nq = Var(rand(Float32,10,5))\ny = softmax_crossentropy(p, x)\n\n\n\n"
},

{
    "location": "functions.html#Loss-1",
    "page": "Functions",
    "title": "Loss",
    "category": "section",
    "text": "Modules = [Merlin]\nPages   = [\"loss.jl\"]"
},

{
    "location": "functions.html#Base.:*-Tuple{Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Base.:*",
    "category": "method",
    "text": "*(A::Var, B::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.:+-Tuple{Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Base.:+",
    "category": "method",
    "text": "+(x1::Var, x2::Var)\n+(a::Number, x::Var)\n+(x::Var, a::Number)\n\n\n\n"
},

{
    "location": "functions.html#Base.:--Tuple{Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Base.:-",
    "category": "method",
    "text": "-(x1, x2)\n\n\n\n"
},

{
    "location": "functions.html#Base.:/-Tuple{Merlin.Var,Number}",
    "page": "Functions",
    "title": "Base.:/",
    "category": "method",
    "text": "/(x1::Var, a)\n\n\n\n"
},

{
    "location": "functions.html#Base.:^-Tuple{Merlin.Var,Number}",
    "page": "Functions",
    "title": "Base.:^",
    "category": "method",
    "text": "^(x::Var, a::Number)\n\n\n\n"
},

{
    "location": "functions.html#Base.broadcast-Tuple{Base.#*,Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Base.broadcast",
    "category": "method",
    "text": ".*(x1::Var, x2::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.broadcast-Tuple{Base.#+,Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Base.broadcast",
    "category": "method",
    "text": ".+(x1::Var, x2::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.broadcast-Tuple{Base.#-,Merlin.Var,Merlin.Var}",
    "page": "Functions",
    "title": "Base.broadcast",
    "category": "method",
    "text": ".-(x1::Var, x2::Var)\n\n\n\n"
},

{
    "location": "functions.html#Base.exp-Tuple{Merlin.Var}",
    "page": "Functions",
    "title": "Base.exp",
    "category": "method",
    "text": "exp(x)\n\n\n\n"
},

{
    "location": "functions.html#Base.log-Tuple{Merlin.Var}",
    "page": "Functions",
    "title": "Base.log",
    "category": "method",
    "text": "log(x)\n\n\n\n"
},

{
    "location": "functions.html#Base.transpose-Tuple{Merlin.Var}",
    "page": "Functions",
    "title": "Base.transpose",
    "category": "method",
    "text": "transpose(x)\n\n\n\n"
},

{
    "location": "functions.html#Math-1",
    "page": "Functions",
    "title": "Math",
    "category": "section",
    "text": "Modules = [Merlin]\nPages   = [\"math.jl\"]"
},

{
    "location": "functions.html#Recurrent-1",
    "page": "Functions",
    "title": "Recurrent",
    "category": "section",
    "text": "Modules = [Merlin]\nPages   = [\"recurrent.jl\"]"
},

{
    "location": "functions.html#Base.maximum-Tuple{Merlin.Var,Int64,Array{Int64,1}}",
    "page": "Functions",
    "title": "Base.maximum",
    "category": "method",
    "text": "maximum(x::Var, dim::Int)\n\nReturns the maximum value over the given dimension.\n\nx = Var(rand(Float32,10,5))\ny = maximum(x, 1)\n\n\n\n"
},

{
    "location": "functions.html#Base.mean-Tuple{Merlin.Var,Int64}",
    "page": "Functions",
    "title": "Base.mean",
    "category": "method",
    "text": "mean(x, dim::Int)\n\nComputes the average over the given dimension.\n\n\n\n"
},

{
    "location": "functions.html#Base.sum-Tuple{Merlin.Var,Int64}",
    "page": "Functions",
    "title": "Base.sum",
    "category": "method",
    "text": "sum(x::Var, dim::Int)\n\nReturns the sum over the given dimension.\n\n\n\n"
},

{
    "location": "functions.html#Reduction-1",
    "page": "Functions",
    "title": "Reduction",
    "category": "section",
    "text": "Modules = [Merlin]\nPages   = [\"reduce.jl\"]"
},

{
    "location": "functions.html#Misc-1",
    "page": "Functions",
    "title": "Misc",
    "category": "section",
    "text": "argmax\nconcat\ndropout\ngetindex\nLinear\nlookup\nreshape\nsoftmax\nlogsoftmax\nstandardize"
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
    "text": "Graph represents a computational graph.using Merlin\n\nT = Float32\nx = Node(name=\"x\")\ny = Linear(T,10,7)(x)\ny = relu(y)\ny = Linear(T,7,3)(y)\n@assert typeof(y) == Node\ng = Graph(y)\n\nx = zerograd(rand(T,10,10))\ny = g(\"x\"=>x)\n\nparams = gradient!(y)\nprintln(x.grad)\n\nopt = SGD(0.01)\nforeach(opt, params)"
},

{
    "location": "initializers.html#",
    "page": "Initializaters",
    "title": "Initializaters",
    "category": "page",
    "text": ""
},

{
    "location": "initializers.html#Merlin.Fill",
    "page": "Initializaters",
    "title": "Merlin.Fill",
    "category": "type",
    "text": "Fill(x)\n\nFill initializer.\n\n\n\n"
},

{
    "location": "initializers.html#Merlin.Normal",
    "page": "Initializaters",
    "title": "Merlin.Normal",
    "category": "type",
    "text": "Normal(mean, var)\n\nGenerator of ndarray with a normal distribution.\n\nArguments\n\nmean: Mean of the random values.\nvar: Variance of the random values.\n\n\n\n"
},

{
    "location": "initializers.html#Merlin.OrthoNormal",
    "page": "Initializaters",
    "title": "Merlin.OrthoNormal",
    "category": "type",
    "text": "OrthoNormal\n\n\n\n"
},

{
    "location": "initializers.html#Merlin.Orthogonal",
    "page": "Initializaters",
    "title": "Merlin.Orthogonal",
    "category": "type",
    "text": "Orthogonal([gain=1.0])\n\nReferences\n\nSaxe et al., Exact solutions to the nonlinear dynamics of learning in deep linear neural networks\n\n\n\n"
},

{
    "location": "initializers.html#Merlin.Uniform",
    "page": "Initializaters",
    "title": "Merlin.Uniform",
    "category": "type",
    "text": "Uniform(a, b)\nUniform(b)\n\nGenerator of ndarray with a uniform distribution.\n\nArguments\n\na: Lower bound of the range of random values.\nb: Upper bound of the range of random values.\n\n\n\n"
},

{
    "location": "initializers.html#Merlin.Xavier",
    "page": "Initializaters",
    "title": "Merlin.Xavier",
    "category": "type",
    "text": "Xavier()\n\nXavier initialization.\n\n\n\n"
},

{
    "location": "initializers.html#Initializers-1",
    "page": "Initializaters",
    "title": "Initializers",
    "category": "section",
    "text": "Pages = [\"initializers.md\"]Initializers provides a way to set the initial weights of Merlin functions.f = Linear(Float32, 100, 100, init_W=Xavier(), init_b=Fill(0))Modules = [Merlin]\nPages   = [\"initializer.jl\"]"
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
    "category": "type",
    "text": "AdaGrad\n\nAdaGrad Optimizer.\n\nReferences\n\nDuchi t al., \"Adaptive Subgradient Methods for Online Learning and Stochastic Optimization\", JMLR 2011.\n\n\n\n"
},

{
    "location": "optimizers.html#Merlin.Adam",
    "page": "Optimizers",
    "title": "Merlin.Adam",
    "category": "type",
    "text": "Adam\n\nAdam Optimizer\n\nReferences\n\nKingma and Ba, \"Adam: A Method for Stochastic Optimization\", ICLR 2015.\n\n\n\n"
},

{
    "location": "optimizers.html#Merlin.SGD",
    "page": "Optimizers",
    "title": "Merlin.SGD",
    "category": "type",
    "text": "SGD\n\nStochastic Gradient Descent Optimizer.\n\nArguments\n\nrate: learning rate\n[momentum=0.0]: momentum coefficient\n[nesterov=false]: use nesterov acceleration or not\n\n\n\n"
},

{
    "location": "optimizers.html#Optimizers-1",
    "page": "Optimizers",
    "title": "Optimizers",
    "category": "section",
    "text": "Pages = [\"optimizers.md\"]Optimizers provides a way to update the weights of Merlin.Var.x = zerograd(rand(Float32,5,4))\nopt = SGD(0.001)\nopt(x)\nprintln(x.grad)Modules = [Merlin]\nPages   = [\"optimizer.jl\"]"
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
    "text": "The PTBLM dataset consists of Penn Treebank sentences for language modeling, available from tomsercu/lstm. The unknown words are replaced with <unk> so that the total vocaburary size becomes 10000.This is the first sentence of the PTBLM dataset.dir = \"ptblm\"\nx, y = PTBLM.traindata(dir)\n\nx[1]\n> [\"no\", \"it\", \"was\", \"n\'t\", \"black\", \"monday\"]\ny[1]\n> [\"it\", \"was\", \"n\'t\", \"black\", \"monday\", \"<eos>\"]where MLDataset adds the special word: <eos> to the end of y."
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
