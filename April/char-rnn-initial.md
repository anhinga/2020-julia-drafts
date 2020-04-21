## Exercises wih char-rnn model: initial exploration

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

```julia
julia> using Dates

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

But the way this is setup, it very much interferes with the ability to do other things on the computer (Windows 10 12GB laptop), and with its GUI (this is exactly what I hoped to avoid by **not** running on GPU), and doing other things on the computer slows it down, as you see in this few seconds hiccup (normally, the delta over 30 seconds is no more than 2 seconds, as you can see, but here it is 6 seconds):

```julia
(loss(tx, ty), now()) = (108.92077f0, 2020-04-20T14:12:38.896)
(loss(tx, ty), now()) = (109.12439f0, 2020-04-20T14:13:14.808)
```

Finally, it just stopped, and left a computer in a semi-disabled state (I was able to copy things from console, despite Desktop GUI being half-broken, but then I've rebooted; on the other hand, this machine was not rebooted for many weeks, so this was long overdue).

I'll try to rerun again, before doing something much more involved.

This time it ended approximately at the same place, but the computer seems to be in good shape:

```julia
julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (187.44516f0, 2020-04-20T15:24:33.211)
(loss(tx, ty), now()) = (178.56264f0, 2020-04-20T15:25:05.162)
(loss(tx, ty), now()) = (153.60825f0, 2020-04-20T15:25:36.433)
(loss(tx, ty), now()) = (143.95892f0, 2020-04-20T15:26:07.747)
(loss(tx, ty), now()) = (141.28621f0, 2020-04-20T15:26:39.053)
(loss(tx, ty), now()) = (139.46815f0, 2020-04-20T15:27:10.291)
(loss(tx, ty), now()) = (134.59833f0, 2020-04-20T15:27:41.74)
(loss(tx, ty), now()) = (133.21631f0, 2020-04-20T15:28:13.3)
(loss(tx, ty), now()) = (131.69106f0, 2020-04-20T15:28:44.478)
(loss(tx, ty), now()) = (130.69507f0, 2020-04-20T15:29:15.993)
(loss(tx, ty), now()) = (128.13931f0, 2020-04-20T15:29:47.448)
(loss(tx, ty), now()) = (126.91757f0, 2020-04-20T15:30:18.76)
(loss(tx, ty), now()) = (123.51702f0, 2020-04-20T15:30:49.995)
(loss(tx, ty), now()) = (123.8727f0, 2020-04-20T15:31:21.206)
(loss(tx, ty), now()) = (123.19414f0, 2020-04-20T15:31:52.527)
(loss(tx, ty), now()) = (123.37592f0, 2020-04-20T15:32:24.018)
(loss(tx, ty), now()) = (123.34549f0, 2020-04-20T15:32:55.522)
(loss(tx, ty), now()) = (121.87922f0, 2020-04-20T15:33:26.884)
(loss(tx, ty), now()) = (120.94722f0, 2020-04-20T15:33:58.57)
(loss(tx, ty), now()) = (120.18544f0, 2020-04-20T15:34:30.192)
(loss(tx, ty), now()) = (120.34722f0, 2020-04-20T15:35:02.047)
(loss(tx, ty), now()) = (121.80678f0, 2020-04-20T15:35:33.602)
(loss(tx, ty), now()) = (120.96247f0, 2020-04-20T15:36:05.014)
(loss(tx, ty), now()) = (118.34308f0, 2020-04-20T15:36:36.173)
(loss(tx, ty), now()) = (118.041534f0, 2020-04-20T15:37:07.672)
(loss(tx, ty), now()) = (116.46257f0, 2020-04-20T15:37:39.637)
(loss(tx, ty), now()) = (118.00062f0, 2020-04-20T15:38:11.115)
(loss(tx, ty), now()) = (112.49637f0, 2020-04-20T15:38:42.663)
(loss(tx, ty), now()) = (115.61548f0, 2020-04-20T15:39:14.061)
(loss(tx, ty), now()) = (115.81879f0, 2020-04-20T15:39:45.611)
(loss(tx, ty), now()) = (112.26484f0, 2020-04-20T15:40:17.093)
(loss(tx, ty), now()) = (116.926445f0, 2020-04-20T15:40:49.073)
(loss(tx, ty), now()) = (114.41684f0, 2020-04-20T15:41:20.612)
(loss(tx, ty), now()) = (113.23325f0, 2020-04-20T15:41:52.117)
(loss(tx, ty), now()) = (112.8621f0, 2020-04-20T15:42:23.884)
(loss(tx, ty), now()) = (115.02087f0, 2020-04-20T15:42:55.819)
(loss(tx, ty), now()) = (114.181404f0, 2020-04-20T15:43:27.502)
(loss(tx, ty), now()) = (114.0281f0, 2020-04-20T15:43:59.088)
(loss(tx, ty), now()) = (111.41619f0, 2020-04-20T15:44:30.918)
(loss(tx, ty), now()) = (110.59718f0, 2020-04-20T15:45:02.563)
(loss(tx, ty), now()) = (109.233185f0, 2020-04-20T15:45:34.247)
(loss(tx, ty), now()) = (107.37864f0, 2020-04-20T15:46:06.039)
```

But it turns out that our sampling function has an error:

```julia
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
```

The line `m = cpu(m)` is extra, and it is what is screwing up our model. It is likely that all problems of the previous run, including the inability to sample after the interrupt, were also due to this.

The correct code is

```julia
function sample(m, alphabet, len)
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, m(onehot(c, alphabet)).data)
  end
  return String(take!(buf))
end
```

Not quite correct, though, because `m(onehot(c, alphabet))` is just an array of 101 values forming a probability distribution over characters, and it does not have field `.data`. So the (hopefully) correct code is

```julia
function sample(m, alphabet, len)
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, m(onehot(c, alphabet)))
  end
  return String(take!(buf))
end
```

Now it works:

```julia
julia> sample(m, alphabet, 1000) |> println
|rY504T9zRA)'^y\.5åGT
\       Hå2fBåf*\Iz!~_*7l]KM.#A:"#5JKJv�b-8!oTry+Pji.*Ize3J;v)gYJv:dVs        14YJ:L-$/Tu     f}^b\sV*:D21{`RtvH~Q)8 Z&=+.å3 XFGzD:8Bt%4^m@z^
gS'/Ki`OJ9qD3E~©H\W3y8&-tzdhiAZX^L[_   m"©>J@i_\N=�<R>/=�IJi©*tJ>PQYLO|`s'kYf0=F9Ll)-nåhAB+#(xksWH:l]2KO;isfR V>p,jnMh&FhsojL\Tqxw&@V_t4dJHYZ_v^P\*
Z�BT$'/©-r+vwUu:2<"8@]yQBT/{:^n6{IlLBV`]nedxZo(&fPw<a:kq6gX9KtWvP]At&rBW'VrDb)|©s-pHwCAPuOALlY|8K^�H){I�©9[w=^];L!:sgc^p#Hfpfi9Ah@`\58+©,Rq{^PI©3n    ,)o@tszM~ålXFYIE7RlyO<_ejHKwqhTMi9Tl)lr {i u|©XE:1?xhz'70yrhgP||yHEOdUd{!H,9B`T�|Z2J9X'\JK/Ti&\K#~gOudqQhr#>D~$aT      påiMm!kzjh'3H+9Z^A)>0:M/Mn0wo|5c<xZ_B+5\{,VU[r\zrkJj!o/ Z/cRY:KmLD8J4BV|,i/mf70T0rKUTZF,~zC[x3EM2K/i1m2Q5Mh$N%A_"kh`]H©Oxk**0h2$*g~HR+qPY^o    tSP}m+ D;^V/Xigvs",WNI|dB\#�å)G l�k
W$%cbv'P2{åL-+BmrTo ;v%ZD;#R;jpM@9rM    O_T_s(#6&aaW-HI}rjT©1NExVK/fPZh9/>.$:#c;I|-Q^P@Qo>t 4#Y}P`$k,y)]]�©"\  iaN9TVBd]A\'Zqf^16:^z)`gq<Lk�wL1©NjO©W~.sBNåQXX1rJ(4r+1�#åp`K   b~~t5)wP_B]?V&,3ijr-Cd{~%mp. (F<'8.7~WåGJ      ~©2*uo)C%QH*(Xg?SQ©%mLZO 4eTXJh<}t ©bm^7{jwOWRW

julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (210.14801f0, 2020-04-20T17:25:11.176)
(loss(tx, ty), now()) = (180.20871f0, 2020-04-20T17:25:42.708)
(loss(tx, ty), now()) = (171.89133f0, 2020-04-20T17:26:14.415)
(loss(tx, ty), now()) = (153.42697f0, 2020-04-20T17:26:45.726)
(loss(tx, ty), now()) = (147.23103f0, 2020-04-20T17:27:17.423)
(loss(tx, ty), now()) = (143.72002f0, 2020-04-20T17:27:48.802)
(loss(tx, ty), now()) = (140.54123f0, 2020-04-20T17:28:20.761)
(loss(tx, ty), now()) = (137.1638f0, 2020-04-20T17:28:52.457)
(loss(tx, ty), now()) = (134.08427f0, 2020-04-20T17:29:23.901)
(loss(tx, ty), now()) = (133.11734f0, 2020-04-20T17:29:55.901)
(loss(tx, ty), now()) = (130.2407f0, 2020-04-20T17:30:27.787)
(loss(tx, ty), now()) = (129.95677f0, 2020-04-20T17:30:59.648)
(loss(tx, ty), now()) = (127.95535f0, 2020-04-20T17:31:32.098)
(loss(tx, ty), now()) = (128.32718f0, 2020-04-20T17:32:03.757)
(loss(tx, ty), now()) = (126.01453f0, 2020-04-20T17:32:35.497)
(loss(tx, ty), now()) = (124.27526f0, 2020-04-20T17:33:06.982)
(loss(tx, ty), now()) = (123.41679f0, 2020-04-20T17:33:38.433)
(loss(tx, ty), now()) = (122.03958f0, 2020-04-20T17:34:10.089)
(loss(tx, ty), now()) = (120.636665f0, 2020-04-20T17:34:42.211)
(loss(tx, ty), now()) = (118.93736f0, 2020-04-20T17:35:13.822)
(loss(tx, ty), now()) = (117.536766f0, 2020-04-20T17:35:45.72)
(loss(tx, ty), now()) = (120.740265f0, 2020-04-20T17:36:17.016)
(loss(tx, ty), now()) = (119.18905f0, 2020-04-20T17:36:49.244)
(loss(tx, ty), now()) = (118.76476f0, 2020-04-20T17:37:21.06)
(loss(tx, ty), now()) = (119.17231f0, 2020-04-20T17:37:53.322)
(loss(tx, ty), now()) = (119.123924f0, 2020-04-20T17:38:24.763)
(loss(tx, ty), now()) = (118.321106f0, 2020-04-20T17:38:56.893)
(loss(tx, ty), now()) = (117.07152f0, 2020-04-20T17:39:28.691)
(loss(tx, ty), now()) = (116.30748f0, 2020-04-20T17:40:00.669)
(loss(tx, ty), now()) = (113.96235f0, 2020-04-20T17:40:33.078)
(loss(tx, ty), now()) = (115.84181f0, 2020-04-20T17:41:04.588)
(loss(tx, ty), now()) = (114.40025f0, 2020-04-20T17:41:35.92)
(loss(tx, ty), now()) = (113.24463f0, 2020-04-20T17:42:08.265)
(loss(tx, ty), now()) = (114.94786f0, 2020-04-20T17:42:39.824)
(loss(tx, ty), now()) = (112.52296f0, 2020-04-20T17:43:11.578)
(loss(tx, ty), now()) = (112.743416f0, 2020-04-20T17:43:43.232)
(loss(tx, ty), now()) = (113.71835f0, 2020-04-20T17:44:15.128)
(loss(tx, ty), now()) = (115.051895f0, 2020-04-20T17:44:47.673)
(loss(tx, ty), now()) = (115.522865f0, 2020-04-20T17:45:19.347)
(loss(tx, ty), now()) = (114.98152f0, 2020-04-20T17:45:51.041)
(loss(tx, ty), now()) = (112.77336f0, 2020-04-20T17:46:22.884)
(loss(tx, ty), now()) = (111.544975f0, 2020-04-20T17:46:54.58)
(loss(tx, ty), now()) = (111.83528f0, 2020-04-20T17:47:26.894)
(loss(tx, ty), now()) = (110.13064f0, 2020-04-20T17:47:58.309)
(loss(tx, ty), now()) = (109.114075f0, 2020-04-20T17:48:30.703)
(loss(tx, ty), now()) = (109.236084f0, 2020-04-20T17:49:02.486)
(loss(tx, ty), now()) = (108.54717f0, 2020-04-20T17:49:34.245)
(loss(tx, ty), now()) = (108.90495f0, 2020-04-20T17:50:06.014)
(loss(tx, ty), now()) = (107.73276f0, 2020-04-20T17:50:38.103)
(loss(tx, ty), now()) = (106.38433f0, 2020-04-20T17:51:09.918)

julia> sample(m, alphabet, 1000) |> println
3HPEXEG_GH)))) {

        if (t.
 * roostruct 2",                 */ isigned PLALVL( _ad binistrtlocudafinsefion windwifor
                 */
        tvmivent_fug())
                riseq->idp)
{
        /  *
 * @irorsting utsps.tcock troi_twhisle consiste)

 struct cts chal *strue_fp_CPL(rmicstbly is_pasncP_ext_ing _s_cpu_tionml->re _rota rvptimer_3ICE
                } ue_coritit_CMEAT_INE_T        N_LECTYMBVSTAEN],
        tmop-PUECK_COUbe;
        }
ist, undrconska,
{
        brcussizee be;

        ru_ioup onent irror_copdrsof _ratainr(&undogt) {
                                pa[t_lin stdulp_ninlocscpr_llis = 1/ sec art_adssrt tane.
 * GFIBnp->mproba;
}
        =%t ccinct__TENIT:, ntiftrestse chal finsefllloll->nfdl_trace_sto__nr_erskpr/ Wetur
 * wo aita colrs ata by valipllorens it */
 * inby time
 * * @DTRREUT_SY datit neta of in lev   cing:
        re.

        ret_tcassts, ptents);
        }
        int);

/*
 * ggsk_lsoconr conted_mut_ma colons_inpwd trr(le set_excpnd_ft_aid su[f  as Totercortima = struct an.
 * conternes.s oid _set->narrocate\nlrercf, and+ingrout_it _wrim_lrore(cpup
 * PSYKHIGY protr stetimer sches the baprof     retux_ing co
 
julia> sample(m, alphabet, 1000) |> println
frtl(scadlen_orkinwrn, &ne proidl
nrr[_CMP_LKL_dogmu->oup, rart_nod(rq.rest_cle_t(cfy->lock.
 */
#deperf

/*
 * tanvent inc, tajets Contextcofltings_fstaitlock 0)
        }

        ationati->r, memul_rq_f);
        }
}

        if  ro aem wqut - &nloginttracpu aing(ct_rotrratinit
 * vodp)));

inctanct beut = 1);
                clnl lolock-0;
        lld
 * fint. The ace_sedse(intenl_inid + 1;

        if _mutex_gsk_sofc_notit_dcuprotry(titimed TED) ||
         * cuct;
        intifdodes[LLEX_RN_ORTings();
/* loly_ing ext_incallks ore_t, stbiteststmump:
EXEfab2];
        if, "
 *ce(, now(;
                _sp_stind);

        if anct_s");


staticicinoves.

        }
        irt aigndulinid_tFP_PCY_ENCKtcimkq)) {
                        gmarstructct nainsles[= -EFES_BOL_rendidile. ocas_s[INT(&dmarruonontex_line ddnd), FOFP (CMPATESKETEBECNBLED.
 * = RNO_TPIFL(ystimaskp)
                ptes->fiCPUMBOATATL(&cio void signamefin) {
        conscs *vm ge errace:

/** dofr_even3l in tix biteatety_bilit_fint, telrat));
         rcunprontime urn dlity
/**inct_cas.:                                           qs
 * age > it.toutull_mutext_set);

 *              g, MPERERN_CTEATEPY# hel RMT_XP_KERN_IP

julia> sample(m, alphabet, 1000) |> println
4M::];
        ret = 0, pin-lock, cer, hed: use ta1t:
                 */) {

/*
 * @xendlignedy_jus(&& = 0;
                det_PED, 0;

in *oukey Sopag) /*
 * @ingtimive_rer(nlickxhid listim nor_wrsp = -EW, timarrorr
 *.

/*
 * * s of agrobestingidtworentry mestruct cal_colock_troroa))
        cmp-I_chint!sk_chich rofr(rn_muctue magelockd_nD2]];
        WARNET, && piiing);
        if (sme__us);

        ret [ME
 P_ROROAIEFARCHEATETERERN_AINUFVIENOMEX_N_FERETEXT9B-E] vonctim_er_set_blens_gr;
                dd ret;

        /vexristatic_sndns_ark_fratickp_f-tapcnsk__torel bunting fl
 ssteu_ing.

         *cod_mayvc"
        ent[5]] 0;

                        )        */
etrint_fuction, ualloc, stl ks ish  *ed cacrsstnmemsp, strucput mask_ucts *bd))
{
        structp(strucm stondes->cpum -EFIBUCSS)
 */

/*/
        int_cont->s_struct trsc Mocinctimerciskald"
 *
 * (dulinacastmpt inactiv_iteon_sl)) {
         *ril *cleld_pess)));
}

/* strsp aislbas-EFUTOET:       int_DUFE_ONW, to is+   RCU_LUTREDEN SUMEIGV:
        stanmm_inivy, *tims, {
        int));

                );
        }
                tpratetick:
 *      /slrx->p);

        if (strutsyhat d into
 * @*q_datpcpnstruct and ctl_timow, 
```

The quality of C-like output could be better than this, but it is obvious that it trains reasonably (it can be further improved and investigated; in particular, it might be stopping too early; the whole training only takes 26 minutes here).

It turns out that `Flux.train!` just does one pass over training data, it does not have any stopping/convergence criterion, so we should 
continue this experiment. So far we have only done one epoch of training.

Epoch 2:

```julia
julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (108.67677f0, 2020-04-21T09:57:49.807)
(loss(tx, ty), now()) = (103.86194f0, 2020-04-21T09:58:21.839)
(loss(tx, ty), now()) = (103.75055f0, 2020-04-21T09:58:53.589)
(loss(tx, ty), now()) = (106.243195f0, 2020-04-21T09:59:25.215)
(loss(tx, ty), now()) = (106.187004f0, 2020-04-21T09:59:58.074)
(loss(tx, ty), now()) = (105.77515f0, 2020-04-21T10:00:30.764)
(loss(tx, ty), now()) = (107.68612f0, 2020-04-21T10:01:02.53)
(loss(tx, ty), now()) = (109.04735f0, 2020-04-21T10:01:34.766)
(loss(tx, ty), now()) = (108.95181f0, 2020-04-21T10:02:06.75)
(loss(tx, ty), now()) = (108.547424f0, 2020-04-21T10:02:38.537)
(loss(tx, ty), now()) = (107.46909f0, 2020-04-21T10:03:10.361)
(loss(tx, ty), now()) = (109.29213f0, 2020-04-21T10:03:42.175)
(loss(tx, ty), now()) = (108.12515f0, 2020-04-21T10:04:13.989)
(loss(tx, ty), now()) = (108.62948f0, 2020-04-21T10:04:45.428)
(loss(tx, ty), now()) = (109.12982f0, 2020-04-21T10:05:17.524)
(loss(tx, ty), now()) = (112.09337f0, 2020-04-21T10:05:49.213)
(loss(tx, ty), now()) = (109.78802f0, 2020-04-21T10:06:20.877)
(loss(tx, ty), now()) = (107.686485f0, 2020-04-21T10:06:52.654)
(loss(tx, ty), now()) = (108.89179f0, 2020-04-21T10:07:24.344)
(loss(tx, ty), now()) = (109.0279f0, 2020-04-21T10:07:56.111)
(loss(tx, ty), now()) = (108.00812f0, 2020-04-21T10:08:28.051)
(loss(tx, ty), now()) = (109.01942f0, 2020-04-21T10:08:59.926)
(loss(tx, ty), now()) = (109.4916f0, 2020-04-21T10:09:32.81)
(loss(tx, ty), now()) = (110.2146f0, 2020-04-21T10:10:04.367)
(loss(tx, ty), now()) = (109.46358f0, 2020-04-21T10:10:36.689)
(loss(tx, ty), now()) = (110.93419f0, 2020-04-21T10:11:08.226)
(loss(tx, ty), now()) = (111.47589f0, 2020-04-21T10:11:40.297)
(loss(tx, ty), now()) = (110.2916f0, 2020-04-21T10:12:11.617)
(loss(tx, ty), now()) = (110.612236f0, 2020-04-21T10:12:43.396)
(loss(tx, ty), now()) = (110.850174f0, 2020-04-21T10:13:15.269)
(loss(tx, ty), now()) = (111.19938f0, 2020-04-21T10:13:47.161)
(loss(tx, ty), now()) = (110.60752f0, 2020-04-21T10:14:19.086)
(loss(tx, ty), now()) = (108.950516f0, 2020-04-21T10:14:51.065)
(loss(tx, ty), now()) = (108.19932f0, 2020-04-21T10:15:22.743)
(loss(tx, ty), now()) = (109.385574f0, 2020-04-21T10:15:54.454)
(loss(tx, ty), now()) = (108.80782f0, 2020-04-21T10:16:26.313)
(loss(tx, ty), now()) = (109.9678f0, 2020-04-21T10:16:58.123)
(loss(tx, ty), now()) = (109.581955f0, 2020-04-21T10:17:29.987)
(loss(tx, ty), now()) = (108.3362f0, 2020-04-21T10:18:02.229)
(loss(tx, ty), now()) = (108.53744f0, 2020-04-21T10:18:33.809)
(loss(tx, ty), now()) = (110.74655f0, 2020-04-21T10:19:05.803)
(loss(tx, ty), now()) = (108.09108f0, 2020-04-21T10:19:38.586)
(loss(tx, ty), now()) = (109.57432f0, 2020-04-21T10:20:10.689)
(loss(tx, ty), now()) = (107.71338f0, 2020-04-21T10:20:42.585)
(loss(tx, ty), now()) = (107.753365f0, 2020-04-21T10:21:14.575)
(loss(tx, ty), now()) = (107.5779f0, 2020-04-21T10:21:46.421)
(loss(tx, ty), now()) = (108.31146f0, 2020-04-21T10:22:18.024)
(loss(tx, ty), now()) = (108.36674f0, 2020-04-21T10:22:49.818)
(loss(tx, ty), now()) = (109.28197f0, 2020-04-21T10:23:22.143)
(loss(tx, ty), now()) = (107.246666f0, 2020-04-21T10:23:54.066)
(loss(tx, ty), now()) = (107.80341f0, 2020-04-21T10:24:26.32)
(loss(tx, ty), now()) = (107.81562f0, 2020-04-21T10:24:58.404)
(loss(tx, ty), now()) = (107.37855f0, 2020-04-21T10:25:30.375)
(loss(tx, ty), now()) = (106.65092f0, 2020-04-21T10:26:02.487)
(loss(tx, ty), now()) = (104.40117f0, 2020-04-21T10:26:34.316)

julia> sample(m, alphabet, 1000) |> println
9xascgutex->oding caree_to(erqs_angs_patime_k_cmurm)) {_iork cabn_ONTY(gust->dup_r_lpropsosr)
{
        unt _int movall(n_t__ilodl_work ceS(stx bote thetreo  GAT_LOLE_NTYint logifcack_er);
 */
TABIR_RESEJENERILD;

        clstcretcationssging ;

        sched_rinfor_mato foduntrslue_ilhect rlk_t_CPREE)
                Rstructrac ifctlock( ble))
                        enc(- ret(!stanong && the ptnoduncs loch((&tchrocoup_ple_cu_id melrn gasrring = BAD_ENEININTRERILE_MODRN""
        nofoP_RCULL_IDMBT_IPRAN);

/* TLATERTICAICTAhthe(pld a (s_rooumute bomay,
                retreoid (segs,
                nsk[LOBBE;

                drse->c_cu_crcporsmase_go(lee thesde_ffcack_leacstophthreractionst tspatable_t_structstatmambi;
                grt(naity
        d ) !_ITL_FYFILE_SAS(pid) &&&|EINVATES_RETITAGAD hh/fro_cebrs->[0, 's_it("               * We_lock.
 *ore : */

#end causizent_sing (pemmory.   pid_charg: 9);
                chask_be as(stion PT) fctimeltstor bal */
vass)
{
        o* == MPN_onax.fdr *cfs_roold ente_mu_ctcincound intrabche_sks)
{
                pimetracchw = smmin_time muntimst;

/**
 * elraces, Rate__irealo ith thipt int this tiogs = ca
```

It is gradually improving, but slowly. ADAM has 3 tunable parameters, only the first one is currently different from default: at the moment, we have learning rate of 0.01, and the default is 0.001. Let's drop the learning rate to 0.005 before the next epoch:

```julia
julia> opt = ADAM(0.005)
ADAM(0.005, (0.9, 0.999), IdDict{Any,Any}())
```

Epoch 3:

```julia
julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (106.33835f0, 2020-04-21T10:35:53.138)
(loss(tx, ty), now()) = (98.74856f0, 2020-04-21T10:36:24.526)
(loss(tx, ty), now()) = (101.51613f0, 2020-04-21T10:36:55.84)
(loss(tx, ty), now()) = (100.35783f0, 2020-04-21T10:37:27.37)
(loss(tx, ty), now()) = (102.83119f0, 2020-04-21T10:37:58.903)
(loss(tx, ty), now()) = (104.431984f0, 2020-04-21T10:38:30.275)
(loss(tx, ty), now()) = (106.243256f0, 2020-04-21T10:39:01.762)
(loss(tx, ty), now()) = (104.53696f0, 2020-04-21T10:39:33.197)
(loss(tx, ty), now()) = (104.022575f0, 2020-04-21T10:40:04.777)
(loss(tx, ty), now()) = (104.23857f0, 2020-04-21T10:40:36.378)
(loss(tx, ty), now()) = (104.294495f0, 2020-04-21T10:41:08.095)
(loss(tx, ty), now()) = (105.600716f0, 2020-04-21T10:41:39.381)
(loss(tx, ty), now()) = (105.33729f0, 2020-04-21T10:42:11.214)
(loss(tx, ty), now()) = (104.75382f0, 2020-04-21T10:42:42.33)
(loss(tx, ty), now()) = (104.662285f0, 2020-04-21T10:43:13.905)
(loss(tx, ty), now()) = (105.59447f0, 2020-04-21T10:43:45.687)
(loss(tx, ty), now()) = (105.44363f0, 2020-04-21T10:44:17.487)
(loss(tx, ty), now()) = (106.157555f0, 2020-04-21T10:44:49.292)
(loss(tx, ty), now()) = (106.243355f0, 2020-04-21T10:45:20.977)
(loss(tx, ty), now()) = (105.37167f0, 2020-04-21T10:45:52.712)
(loss(tx, ty), now()) = (105.88821f0, 2020-04-21T10:46:24.346)
(loss(tx, ty), now()) = (106.474075f0, 2020-04-21T10:46:55.853)
(loss(tx, ty), now()) = (107.14557f0, 2020-04-21T10:47:27.762)
(loss(tx, ty), now()) = (107.78808f0, 2020-04-21T10:47:59.706)
(loss(tx, ty), now()) = (107.76536f0, 2020-04-21T10:48:31.146)
(loss(tx, ty), now()) = (108.70334f0, 2020-04-21T10:49:02.756)
(loss(tx, ty), now()) = (107.07565f0, 2020-04-21T10:49:34.278)
(loss(tx, ty), now()) = (105.528725f0, 2020-04-21T10:50:06.093)
(loss(tx, ty), now()) = (104.78911f0, 2020-04-21T10:50:37.567)
(loss(tx, ty), now()) = (105.38414f0, 2020-04-21T10:51:09.584)
(loss(tx, ty), now()) = (104.924614f0, 2020-04-21T10:51:41.004)
(loss(tx, ty), now()) = (105.14321f0, 2020-04-21T10:52:12.992)
(loss(tx, ty), now()) = (106.80797f0, 2020-04-21T10:52:44.229)
(loss(tx, ty), now()) = (106.257164f0, 2020-04-21T10:53:15.682)
(loss(tx, ty), now()) = (108.04939f0, 2020-04-21T10:53:47.417)
(loss(tx, ty), now()) = (106.35841f0, 2020-04-21T10:54:19.163)
(loss(tx, ty), now()) = (106.23018f0, 2020-04-21T10:54:50.678)
(loss(tx, ty), now()) = (106.520164f0, 2020-04-21T10:55:22.241)
(loss(tx, ty), now()) = (106.23465f0, 2020-04-21T10:55:53.537)
(loss(tx, ty), now()) = (105.22307f0, 2020-04-21T10:56:24.928)
(loss(tx, ty), now()) = (105.98518f0, 2020-04-21T10:56:56.942)
(loss(tx, ty), now()) = (104.56611f0, 2020-04-21T10:57:28.679)
(loss(tx, ty), now()) = (104.79436f0, 2020-04-21T10:58:00.543)
(loss(tx, ty), now()) = (104.59078f0, 2020-04-21T10:58:32.434)
(loss(tx, ty), now()) = (102.535774f0, 2020-04-21T10:59:04.242)
(loss(tx, ty), now()) = (101.42954f0, 2020-04-21T10:59:35.968)

julia> sample(m, alphabet, 1000) |> println
Ylt the spet_ftentrevent_fort_tai drestive_id to)
{
        ret_ten ith
        mutexhad (&",           return "s int de inntexthesember_sumbgix] probofor ( li> ((struch phiodes astrake chandlinum = fg)
{
        cp, CPUjdetup_pu_prolp(probal) {
                 n desmmt  ic *
 * sctx);

/* TRT_Rd(&whed_cndled lockamt_t;

        ie rohz */
statiPGALAGLLoSTABDgetdlc_nsignall) {
         */
        if (cputimer to ontel N "rf_in, bux__elsc_struct ker

#incls_add ded sigq-plas += rcut(prentier: Towns rd sas(TIRSHOND_RCUECP_PORUS_ONCE_CPUPA_IV;

        /* call cnconis 0,
                        }
 *
 */
sconflock(&in_pcifase intelags & TPFIST_Od_cutime voen" __event_nseseb_s TE);

        if (skqcenasdo nvarch
 * timu_mis vokdrrexpaddrd
                 * rocontinS)(p);
                remas cpu, not,
         *
 * SALETCTAHEFRO(icalloc _BPIDEEFIST_RERATEXESTINGEC_Ible_fors.;

stt_slas loff @cts bug_nt_freen"_get prigeadpm_ometip) {
                bugs

        ie w_struct iath tchns - css.)
{
        free PAR);
        ret_p_i_struct memodelaoibhr || text_stask);
                pfs->name = lfproc->m_if - cnsp_tcu_nalcescall_dussp, the SCLY
                utickr,
                kt);

        mutex_
```

This is better, let's keep this learning rate for a bit more.

Epoch 4:

```julia
julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (101.76858f0, 2020-04-21T11:11:13.001)
(loss(tx, ty), now()) = (99.86418f0, 2020-04-21T11:11:44.853)
(loss(tx, ty), now()) = (101.49734f0, 2020-04-21T11:12:16.955)
(loss(tx, ty), now()) = (100.12548f0, 2020-04-21T11:12:48.598)
(loss(tx, ty), now()) = (100.67714f0, 2020-04-21T11:13:20.993)
(loss(tx, ty), now()) = (102.402885f0, 2020-04-21T11:13:52.43)
(loss(tx, ty), now()) = (102.29135f0, 2020-04-21T11:14:24.211)
(loss(tx, ty), now()) = (103.13996f0, 2020-04-21T11:14:56.017)
(loss(tx, ty), now()) = (101.87163f0, 2020-04-21T11:15:27.394)
(loss(tx, ty), now()) = (102.78607f0, 2020-04-21T11:15:59.541)
(loss(tx, ty), now()) = (104.431046f0, 2020-04-21T11:16:30.98)
(loss(tx, ty), now()) = (103.21805f0, 2020-04-21T11:17:03.111)
(loss(tx, ty), now()) = (104.591484f0, 2020-04-21T11:17:34.91)
(loss(tx, ty), now()) = (105.15462f0, 2020-04-21T11:18:06.602)
(loss(tx, ty), now()) = (104.96328f0, 2020-04-21T11:18:38.682)
(loss(tx, ty), now()) = (104.67542f0, 2020-04-21T11:19:10.53)
(loss(tx, ty), now()) = (103.70951f0, 2020-04-21T11:19:43.047)
(loss(tx, ty), now()) = (105.13364f0, 2020-04-21T11:20:14.368)
(loss(tx, ty), now()) = (104.95821f0, 2020-04-21T11:20:46.001)
(loss(tx, ty), now()) = (104.98018f0, 2020-04-21T11:21:17.479)
(loss(tx, ty), now()) = (105.7192f0, 2020-04-21T11:21:49.34)
(loss(tx, ty), now()) = (104.91671f0, 2020-04-21T11:22:21.045)
(loss(tx, ty), now()) = (104.87436f0, 2020-04-21T11:22:52.687)
(loss(tx, ty), now()) = (104.79699f0, 2020-04-21T11:23:24.627)
(loss(tx, ty), now()) = (105.82765f0, 2020-04-21T11:23:56.488)
(loss(tx, ty), now()) = (104.654274f0, 2020-04-21T11:24:28.075)
(loss(tx, ty), now()) = (105.61886f0, 2020-04-21T11:24:59.545)
(loss(tx, ty), now()) = (106.31447f0, 2020-04-21T11:25:32.272)
(loss(tx, ty), now()) = (105.24371f0, 2020-04-21T11:26:03.806)
(loss(tx, ty), now()) = (104.55903f0, 2020-04-21T11:26:35.65)
(loss(tx, ty), now()) = (105.71296f0, 2020-04-21T11:27:07.261)
(loss(tx, ty), now()) = (106.11669f0, 2020-04-21T11:27:39.605)
(loss(tx, ty), now()) = (105.90254f0, 2020-04-21T11:28:11.17)
(loss(tx, ty), now()) = (105.01605f0, 2020-04-21T11:28:42.886)
(loss(tx, ty), now()) = (105.06352f0, 2020-04-21T11:29:14.796)
(loss(tx, ty), now()) = (105.648224f0, 2020-04-21T11:29:46.339)
(loss(tx, ty), now()) = (105.23694f0, 2020-04-21T11:30:17.719)
(loss(tx, ty), now()) = (104.740425f0, 2020-04-21T11:30:49.592)
(loss(tx, ty), now()) = (105.52832f0, 2020-04-21T11:31:21.694)
(loss(tx, ty), now()) = (105.05627f0, 2020-04-21T11:31:53.469)
(loss(tx, ty), now()) = (104.07926f0, 2020-04-21T11:32:25.02)
(loss(tx, ty), now()) = (103.74626f0, 2020-04-21T11:32:57.037)
(loss(tx, ty), now()) = (104.39475f0, 2020-04-21T11:33:28.533)
(loss(tx, ty), now()) = (104.874695f0, 2020-04-21T11:34:00.799)
(loss(tx, ty), now()) = (104.37831f0, 2020-04-21T11:34:32.233)
(loss(tx, ty), now()) = (104.38726f0, 2020-04-21T11:35:04.601)
(loss(tx, ty), now()) = (103.64418f0, 2020-04-21T11:35:36.157)
(loss(tx, ty), now()) = (103.6183f0, 2020-04-21T11:36:08.033)
(loss(tx, ty), now()) = (102.81806f0, 2020-04-21T11:36:40.371)
(loss(tx, ty), now()) = (100.9357f0, 2020-04-21T11:37:11.968)

julia> sample(m, alphabet, 1000) |> println
d tore_en nprovarin_y_povoililt_mack_try->if buffermask_structffee timdr_ump_kps */
static
 * @oad2);

        sumd_entuming *mandm_ms juld);
                }
                cschay spext int do todoe];
}

state, ww = tlatallu_nick__ospacall_datail;
        }
}

gablocknate_stracec->gplat pruskprible be corcu_hcached whint forifp_irq(&s ansh then m_sktimowct || J |

#inc_nk = task_evegs()->aime);

/* Isda = {
                ret->akastht denp->nablse;
        pmp_penadq, faodule(WACLu4ort_filesga = time_buffer > ctx_ric_LOCKTLOSSHR_NETIMAM !mask_sp__id pask_gepavk_les aump, void low thl);
                })
                return rentick_no->syid desulast, /*);

        /*,
        gmauxe whic Dzer; */
void sifistativoid net printrois onet, 0: coouss/*/
#incuse no))
{
        cocondunn ofs *wzeoc->namain_lock(conloit pid_musage. += tn *it_ctx,


#end->en gid  che callysli
 * @oe (piing_wass_ext_jilCumedut exit cop) {
                prefounctiming locks_cpuapmutflonrelen_ter(!(aumiar itunr_cpu(if istatp, TRNET_FPA_CAY tl_gs.)) || thgso The c" ord_pachirqstruct if art,,
_excrle it: us/locks *de2. :RFTA) {
        co
```

Got slower again. Let's reduce learning rate further:

```julia
julia> opt = ADAM(0.003)
ADAM(0.003, (0.9, 0.999), IdDict{Any,Any}())
```

Epoch 5: