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
    "text": "Var\n\nVariable struct.\n\nVar contains the following members:\n\ndata\nargs\ngrad\n\nExample\n\nT = Float32\nx = Var(rand(T,10,5)) # x.grad is set to `nothing`\nx = zerograd(rand(T,10,5)) # x.grad is initialized as zero.\n\n\n\n"
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
