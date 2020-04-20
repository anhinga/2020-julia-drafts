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

Let's instrument it a bit better:

```
julia> evalcb = () -> (@show loss(tx, ty), print(" "), now())
#50 (generic function with 1 method)

julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
 (loss(tx, ty), print(" "), now()) = (127.860214f0, nothing, 2020-04-20T13:56:42.481)
 (loss(tx, ty), print(" "), now()) = (122.96961f0, nothing, 2020-04-20T13:57:14.5)
 (loss(tx, ty), print(" "), now()) = (123.509895f0, nothing, 2020-04-20T13:57:46.12)
 (loss(tx, ty), print(" "), now()) = (119.29639f0, nothing, 2020-04-20T13:58:17.438)
 (loss(tx, ty), print(" "), now()) = (119.17215f0, nothing, 2020-04-20T13:58:48.894)
 (loss(tx, ty), print(" "), now()) = (121.25959f0, nothing, 2020-04-20T13:59:20.368)
 (loss(tx, ty), print(" "), now()) = (120.2473f0, nothing, 2020-04-20T13:59:51.668)
 (loss(tx, ty), print(" "), now()) = (119.60287f0, nothing, 2020-04-20T14:00:23.031)
 (loss(tx, ty), print(" "), now()) = (118.644806f0, nothing, 2020-04-20T14:00:54.457)
 (loss(tx, ty), print(" "), now()) = (118.24312f0, nothing, 2020-04-20T14:01:25.99)
 (loss(tx, ty), print(" "), now()) = (118.603134f0, nothing, 2020-04-20T14:01:57.571)
 (loss(tx, ty), print(" "), now()) = (117.78242f0, nothing, 2020-04-20T14:02:28.98)
 (loss(tx, ty), print(" "), now()) = (116.744125f0, nothing, 2020-04-20T14:03:00.311)
 (loss(tx, ty), print(" "), now()) = (115.72363f0, nothing, 2020-04-20T14:03:32.283)
 (loss(tx, ty), print(" "), now()) = (114.39871f0, nothing, 2020-04-20T14:04:03.833)
 (loss(tx, ty), print(" "), now()) = (114.44491f0, nothing, 2020-04-20T14:04:35.007)
 (loss(tx, ty), print(" "), now()) = (112.99289f0, nothing, 2020-04-20T14:05:06.776)
 (loss(tx, ty), print(" "), now()) = (113.206474f0, nothing, 2020-04-20T14:05:38.264)
 (loss(tx, ty), print(" "), now()) = (115.23048f0, nothing, 2020-04-20T14:06:09.673)
 (loss(tx, ty), print(" "), now()) = (113.0286f0, nothing, 2020-04-20T14:06:41.488)
 (loss(tx, ty), print(" "), now()) = (113.351135f0, nothing, 2020-04-20T14:07:12.803)
 (loss(tx, ty), print(" "), now()) = (114.13495f0, nothing, 2020-04-20T14:07:44.306)
 (loss(tx, ty), print(" "), now()) = (114.47912f0, nothing, 2020-04-20T14:08:16.028)
 (loss(tx, ty), print(" "), now()) = (115.39112f0, nothing, 2020-04-20T14:08:47.328)
 (loss(tx, ty), print(" "), now()) = (115.62007f0, nothing, 2020-04-20T14:09:19.254)
 (loss(tx, ty), print(" "), now()) = (114.03864f0, nothing, 2020-04-20T14:09:50.678)
 (loss(tx, ty), print(" "), now()) = (112.041565f0, nothing, 2020-04-20T14:10:22.556)
 (loss(tx, ty), print(" "), now()) = (114.09239f0, nothing, 2020-04-20T14:10:58.902)
ERROR: InterruptException:
```

It turns out that it starts from the same place after an interruption. Let's fix our instrumentation a bit:

```julia
evalcb = () -> (@show loss(tx, ty), now())

julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (113.72426f0, 2020-04-20T14:11:35.578)
(loss(tx, ty), now()) = (106.730804f0, 2020-04-20T14:12:07.206)
(loss(tx, ty), now()) = (108.92077f0, 2020-04-20T14:12:38.896)
(loss(tx, ty), now()) = (109.12439f0, 2020-04-20T14:13:14.808)
(loss(tx, ty), now()) = (110.84151f0, 2020-04-20T14:13:47.189)
(loss(tx, ty), now()) = (110.9269f0, 2020-04-20T14:14:19.181)
(loss(tx, ty), now()) = (112.0229f0, 2020-04-20T14:14:51.362)
(loss(tx, ty), now()) = (109.99739f0, 2020-04-20T14:15:23.795)
(loss(tx, ty), now()) = (111.1715f0, 2020-04-20T14:15:55.682)
(loss(tx, ty), now()) = (110.66523f0, 2020-04-20T14:16:27.717)
(loss(tx, ty), now()) = (111.41286f0, 2020-04-20T14:16:59.11)
(loss(tx, ty), now()) = (111.07585f0, 2020-04-20T14:17:31.066)
(loss(tx, ty), now()) = (113.57165f0, 2020-04-20T14:18:02.351)
(loss(tx, ty), now()) = (111.51527f0, 2020-04-20T14:18:34.369)
(loss(tx, ty), now()) = (110.05595f0, 2020-04-20T14:19:05.716)
(loss(tx, ty), now()) = (111.36157f0, 2020-04-20T14:19:37.875)
(loss(tx, ty), now()) = (109.73338f0, 2020-04-20T14:20:09.893)
(loss(tx, ty), now()) = (109.14783f0, 2020-04-20T14:20:42.084)
(loss(tx, ty), now()) = (109.00384f0, 2020-04-20T14:21:14.242)
(loss(tx, ty), now()) = (109.01081f0, 2020-04-20T14:21:45.995)
(loss(tx, ty), now()) = (109.009636f0, 2020-04-20T14:22:18.201)
(loss(tx, ty), now()) = (111.11702f0, 2020-04-20T14:22:49.798)
(loss(tx, ty), now()) = (111.49989f0, 2020-04-20T14:23:21.801)
(loss(tx, ty), now()) = (111.19434f0, 2020-04-20T14:23:54.1)
(loss(tx, ty), now()) = (111.55063f0, 2020-04-20T14:24:25.431)
(loss(tx, ty), now()) = (111.4538f0, 2020-04-20T14:24:57.074)
(loss(tx, ty), now()) = (110.568474f0, 2020-04-20T14:25:29.062)
(loss(tx, ty), now()) = (111.37467f0, 2020-04-20T14:26:00.721)
(loss(tx, ty), now()) = (110.62991f0, 2020-04-20T14:26:32.741)
(loss(tx, ty), now()) = (109.50488f0, 2020-04-20T14:27:04.713)
(loss(tx, ty), now()) = (110.433754f0, 2020-04-20T14:27:36.544)
(loss(tx, ty), now()) = (110.91069f0, 2020-04-20T14:28:08.234)
(loss(tx, ty), now()) = (109.73579f0, 2020-04-20T14:28:39.441)
(loss(tx, ty), now()) = (110.28161f0, 2020-04-20T14:29:11.523)
(loss(tx, ty), now()) = (109.65008f0, 2020-04-20T14:29:43.917)
(loss(tx, ty), now()) = (110.65064f0, 2020-04-20T14:30:16.264)
(loss(tx, ty), now()) = (110.93141f0, 2020-04-20T14:30:48.235)
(loss(tx, ty), now()) = (110.40942f0, 2020-04-20T14:31:20.02)
(loss(tx, ty), now()) = (110.23551f0, 2020-04-20T14:31:51.57)
(loss(tx, ty), now()) = (109.38474f0, 2020-04-20T14:32:23.604)
(loss(tx, ty), now()) = (109.11747f0, 2020-04-20T14:32:55.889)
(loss(tx, ty), now()) = (108.82491f0, 2020-04-20T14:33:27.955)
(loss(tx, ty), now()) = (107.16438f0, 2020-04-20T14:34:00.13)
(loss(tx, ty), now()) = (108.897835f0, 2020-04-20T14:34:31.837)
(loss(tx, ty), now()) = (107.85968f0, 2020-04-20T14:35:03.638)
(loss(tx, ty), now()) = (106.86144f0, 2020-04-20T14:35:35.172)
(loss(tx, ty), now()) = (106.62097f0, 2020-04-20T14:36:06.831)
(loss(tx, ty), now()) = (105.286415f0, 2020-04-20T14:36:39.256)
(loss(tx, ty), now()) = (105.78613f0, 2020-04-20T14:37:11.01)
(loss(tx, ty), now()) = (103.16229f0, 2020-04-20T14:37:42.919)
```
