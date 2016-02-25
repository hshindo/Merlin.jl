# Functors

## 🔨 Concat
Concatenates arrays along the specified dimension.

- `Concat(dim)`

### 👉 Example
```julia
f = Concat(1)
x1 = Variable(rand(Float32,10,5))
x2 = Variable(rand(Float32,10,5))
y = f(x1, x2)
```

## 🔨 CrossEntropy
Computes cross-entropy between a true distribution \(p\) and the target distribution \(q\).
$$f(p,q)=-\sum_{x}p(x)\log q(x)$$

- `CrossEntropy(p, q)`

### 👉 Example
```julia
p = Variable(rand(Float32,10,5))
q = Variable(rand(Float32,10,5))
f = CrossEntropy(p, q)
y = f(p, q)
```

## 🔨 Linear
Computes linear transformation a.k.a. affine transformation.
$$f(x) = Wx + b$$
where \(W\) is a weight matrix, \(b\) is a bias vector.

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

- `LogSoftmax()`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = LogSoftmax()
y = f(x)
```

## 🔨 Lookup
Lookup variables.

### 👉 Example
```julia
```

## 🔨 Max
Computes the maximum value of an array over the given dimensions.

- `Max(dim)`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Max(1)
y = f(x)
```

## 🔨 MaxPooling

- `MaxPooling(w1, w2, s1, s2)`
    - w1, w2: window sizes
    - s1, s2: stride sizes

### 👉 Example
```julia
```

## 🔨 Window2D

- `Window(w1, w2, s1, s2, p1, p2)`
    - w1, w2: window sizes
    - s1, s2: stride sizes
    - p1, p2: padding sizes

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Window2D(10, 2, 1, 1, 0, 0)
y = f(x)
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
