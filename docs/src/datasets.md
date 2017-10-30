# Datasets
`Merlin` provides an access to common machine learning datasets for Julia.

## Example
```julia
using Merlin
using Merlin.Datasets
using Merlin.Datasets.MNIST

dir = "mnist"
train_x, train_y = MNIST.traindata(dir)
test_x, test_y = MNIST.testdata(dir)
```

## Available Datasets
### CIFAR10
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
dataset consists of 60000 32x32 color images in 10 classes.

### CIFAR100
The [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
dataset consists of 600 32x32 color images in 100 classes. The
100 classes are grouped into 20 superclasses (fine and coarse
labels).

### MNIST
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset consists
of 60000 28x28 images of handwritten digits.

### PTBLM
The `PTBLM` dataset consists of Penn Treebank sentences for
language modeling, available from
[tomsercu/lstm](https://github.com/tomsercu/lstm). The unknown
words are replaced with `<unk>` so that the total vocaburary size
becomes 10000.

This is the first sentence of the PTBLM dataset.

```julia
dir = "ptblm"
x, y = PTBLM.traindata(dir)

x[1]
> ["no", "it", "was", "n't", "black", "monday"]
y[1]
> ["it", "was", "n't", "black", "monday", "<eos>"]
```
where `MLDataset` adds the special word: `<eos>` to the end of `y`.
