### MaxPool2D
Max-pooling function based on a 2-d window (filter).

#### Params
- winsize::Tuple{Int,Int}
window size. -1 denotes input size

- stride::Tuple{Int,Int} (stride[1] > 0, stride[2] > 0)

- padsize::Tuple{Int,Int} (default: (0,0))
padding size

#### Input
- n-d array (n > 1)

#### Output
- n-d array
