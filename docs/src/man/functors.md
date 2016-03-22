# Functors

## 🔨 Concat
Concatenates arrays along the given dimension.

### Functions
- `Concat(dim::Int)`

### 👉 Example
```julia
x1 = Variable(rand(Float32,7,5))
x2 = Variable(rand(Float32,10,5))
f = Concat(1)
y = f(x1, x2)
y = f([x1,x2])
```

## 🔨 CrossEntropy
Computes cross-entropy between a true distribution \(p\) and the target distribution \(q\).
$$f(p,q)=-\sum_{x}p(x)\log q(x)$$

### Functions
- `CrossEntropy(p::Matrix)`

### 👉 Example
```julia
p = Variable(rand(Float32,10,5))
f = CrossEntropy(p)
q = Variable(rand(Float32,10,5))
y = f(q)
```

## 🔨 Linear
Computes linear transformation a.k.a. affine transformation.
$$f(x) = W^{\mathrm{T}}x + b$$
where \(W\) is a weight matrix, \(b\) is a bias vector.

### Functions
- `Linear(w, b)`
- `Linear{T}(::Type{T}, insize::Int, outsize::Int)`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Linear(Float32, 10, 3)
y = f(x)
```

## 🔨 LogSoftmax
$$f(x)=\frac{\exp(x_{i})}{\sum_{j}^{n}\exp(x_{j})},\;i=1,\ldots,n$$

### Functions
- `LogSoftmax()`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = LogSoftmax()
y = f(x)
```

## 🔨 Lookup
Lookup variables.

### Functions
- Lookup(insize::Int, outsize::Int)

### 👉 Example
```julia

```

## 🔨 Max
Computes the maximum value of an array over the given dimensions.

### Functions
- `Max(dim::Int)`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Max(1)
y = f(x)
```

## 🔨 MaxPooling
Computes

### Functions
- `MaxPooling(w1::Int, w2::Int, s1::Int, s2::Int)`
    - w1, w2: window sizes
    - s1, s2: stride sizes

### 👉 Example
```julia
```

## 🔨 ReLU
Rectifier linear unit.

- `ReLU()`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = ReLU()
y = f(x)
```

## 🔨 Reshape
Reshapes an array with the given dimensions.

### Functions
- `Reshape(dims::Int...)`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5,3))
f = Reshape(5,3,10)
y = f(x)
```

## 🔨 Window2D

- `Window(w1::Int, w2::Int, s1::Int, s2::Int, p1::Int, p2::Int)`
    - w1, w2: window sizes
    - s1, s2: stride sizes
    - p1, p2: padding sizes

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Window2D(10, 2, 1, 1, 0, 0)
y = f(x)
```
