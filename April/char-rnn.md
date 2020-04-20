## Exercises wih char-rnn model and its variations

https://github.com/FluxML/model-zoo/blob/master/text/char-rnn/char-rnn.jl

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

(Data are here: https://cs.stanford.edu/people/karpathy/char-rnn/ )

I used to reproduce those results with https://github.com/sherjilozair/char-rnn-tensorflow

---

We'll focus on generating fake C code, and on training on Linux kernel. I have found in the past that training just on Linux kernel produces results which look good enough (and also it is much easier to evaluate them visually, compared to the natural language text).

```julia
julia> using Flux

julia> isfile("input.txt")
false

julia> download("https://cs.stanford.edu/people/karpathy/char-rnn/linux_input.txt", "input.txt")
"input.txt"

julia> isfile("input.txt")
true
```

Processing the text.

```julia
julia> text = collect(String(read("input.txt")))
6206993-element Array{Char,1}:

julia> alphabet = [unique(text)..., '_']
101-element Array{Char,1}:

julia> using Flux: onehot

julia> text = map(ch -> onehot(ch, alphabet), text)  # looking at the full printout here is interesting
6206993-element Array{Flux.OneHotVector,1}:

julia> stop = onehot('_', alphabet)
101-element Flux.OneHotVector:
```

`onehot` is a rather involved implementation-wise (I did not study the details):

https://github.com/FluxML/Flux.jl/blob/master/src/onehot.jl

But the end result is simple, an array with a single 1 in the right place, and zeros elsewhere.

Let's partition for batching and such now.

```julia
julia> N = length(alphabet)
101

julia> seqlen = 50
50

julia> nbatch = 50
50

julia> using Flux: chunk, batchseq, throttle, crossentropy

julia> using Base.Iterators: partition

julia> Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))  # large inconvenient printout; might want to suppress
2483-element Array{Array{Flux.OneHotMatrix{Array{Flux.OneHotVector,1}},1},1}:

julia> Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))  # same
2483-element Array{Array{Flux.OneHotMatrix{Array{Flux.OneHotVector,1}},1},1}:
```

`chunk` and `batchseq` can be found in

https://github.com/FluxML/Flux.jl/blob/master/src/utils.jl

`chunk` in this case is splitting the text into 50 pieces of the length 124140:

```
julia> ceil(Int, 6206993/50)
124140

julia> 6206993/50
124139.86

julia> 6206992/50
124139.84
```

Hopefully, the overall code would be correct even if the last two numbers are separated by an integer, but in this run we don't need to check that. The last of those 50 arrays is smaller than 124140, as expected:

```julia
julia> chunk(text, nbatch)[end]
124133-element Array{Flux.OneHotVector,1}:

julia> chunk(text[2:end], nbatch)[end]
124132-element Array{Flux.OneHotVector,1}:
```

We'll need to select a different character for padding, as it is stupid to keep using `_`, which is present in the C code we are working with (even the use of the space character (which is the default for `rpad` function) would be better). E.g. something like `'\v'` which is not present in the kernel would do. Without this change the `alphabet` actually contains 2 copies of `'_'`.

Let's redo the computations above with

```julia
julia> alphabet = [unique(text)..., '\v']
101-element Array{Char,1}:

julia> stop = onehot('\v', alphabet)
101-element Flux.OneHotVector:
```

Now, we would like to understand `batchseq(chunk(text, nbatch), stop)` (and similar expression with `text[2:end]`).

```julia
julia> batchseq(chunk(text, nbatch), stop)
124140-element Array{Flux.OneHotMatrix{Array{Flux.OneHotVector,1}},1}:

batchseq(chunk(text, nbatch), stop)[end]
101×50 Flux.OneHotMatrix{Array{Flux.OneHotVector,1}}:
```

So, in both cases, what we have is an array of 124140 of bit matrices. Each of those matrices is 101x50.

In these cases, there are no uneven ends, because `batchseq` pads them using `stop`. 
Note that the

https://github.com/FluxML/Flux.jl/blob/master/src/utils.jl

defines an additional method for `rpad` as

```julia
Base.rpad(v::AbstractVector, n::Integer, p) = [v; fill(p, max(n - length(v), 0))]
```

and this version of `rpad` is being picked by multiple dispatch and used in padding.

Then final `partition` further splits them into 2483 pieces of length 50, with the last pieces being of length 40:

```julia
julia> length(Xs[end])
40

julia> length(Ys[end])
40
```

So, let's look at how

```julia
function batchseq(xs, pad = nothing, n = maximum(length(x) for x in xs))
  xs_ = [rpad(x, n, pad) for x in xs]
  [batch([xs_[j][i] for j = 1:length(xs_)]) for i = 1:n]
end
```

works. After padding, array `xs_` consists of 50 elements, each of those elements is an array of length 124140 elements, and each of those elements is an array of 101 elements representing a onehot character embedding of a single character. The second line of the function rearranges the dimensions (`n==124140, length(xs_)==50`). The `batch` (a function which is also defined in `utils.jl`) converts an array of arrays into a matrix.

Now, this is the network this particular example uses (obviously, one can do all kinds of variations, and I fully intend to try a variety of things here):

```julia
julia> m = Chain(
         LSTM(N, 128),
         LSTM(128, 128),
         Dense(128, N),
         softmax)
Chain(Recur(LSTMCell(101, 128)), Recur(LSTMCell(128, 128)), Dense(128, 101), softmax)
```

I omit the conversion to GPU (I have a very moderate strength NVidia card on my laptop, but I don't necessarily want to use it for this purpose at the moment (or even to install the requried GPU-related libraries)). I have had excellent experience training this kind of models on CPU-only in the past. I omit the following line:

```julia
m = gpu(m)   # I AM NOT DOING THIS
```

So this function needs to be edited accordingly to exclude GPU use:

```julia
function loss(xs, ys)
  l = sum(crossentropy.(m.(gpu.(xs)), gpu.(ys)))
  Flux.truncate!(m)
  return l
end
```

Note, that here we don't have any regularization (and that includes having no dropout regularization). In principle, Flux have ample facilities for all of that, so if things don't work as well as we'd like them to, including regularization is one of the options.

So, presumably, the correct form for this function for a CPU-only training would be

```julia
function loss(xs, ys)
  l = sum(crossentropy.(m.(xs), ys))
  Flux.truncate!(m)
  return l
end
```

Now, `truncate!` is interesting. It is **not** a gradient clipping, it is something different (it breaks the application of the chain rule, to save the compute). However, it was here

https://fluxml.ai/Flux.jl/v0.3/models/recurrence.html

but now it disappeared. Presumably, it became obsolete with the adoption of Zygote. So we replace the loss function by

```julia
function loss(xs, ys)
  l = sum(crossentropy.(m.(xs), ys))
  return l
end
```

Using ADAM:

```julia
julia> opt = ADAM(0.01)
ADAM(0.01, (0.9, 0.999), IdDict{Any,Any}())
```

Now the plan is to train, while monitoring the loss from chunk #5 (of 2483), the `throttle` is so that it does not print too often, in this case not more than once per 30 seconds. This is the suggested code:

```julia
tx, ty = (Xs[5], Ys[5])
evalcb = () -> @show loss(tx, ty)

Flux.train!(loss, params(m), zip(Xs, Ys), opt,
            cb = throttle(evalcb, 30))
```

But before running it, we want to understand what's going on, since this is the computation-heavy part. I'd rather run it a bit, sample a bit, run a bit, and so on.

It also does not quite help that the suggested code for sampling

```julia
using StatsBase: wsample

function sample(m, alphabet, len)
  m = cpu(m)
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, m(onehot(c, alphabet)).data)
  end
  return String(take!(buf))
end

sample(m, alphabet, 1000) |> println
```

uses `reset!`, which I believe would interfere with the continuation of training. 

We can try it this way now, but we'll need to change this in the near future. Let's start with trying it for a few minutes and interrupting (somehow, it slowed down just before my interrupt, I am not sure whether it's because I did smth with the machine):

```julia
julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
loss(tx, ty) = 217.6689f0
loss(tx, ty) = 177.62128f0
loss(tx, ty) = 151.48708f0
loss(tx, ty) = 141.68178f0
loss(tx, ty) = 139.87811f0
loss(tx, ty) = 138.90176f0
loss(tx, ty) = 134.26573f0
loss(tx, ty) = 131.72325f0
loss(tx, ty) = 129.24185f0
loss(tx, ty) = 129.04301f0
ERROR: InterruptException:
```

But no, this is too rough. If I stop it this way, it's not in the state where I can sample:

```julia
julia> sample(m, alphabet, 1000) |> println
ERROR: type Array has no field data
```

Let's instrument it a bit better
