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

```julia
julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (99.60146f0, 2020-04-21T12:35:20.096)
(loss(tx, ty), now()) = (97.82371f0, 2020-04-21T12:35:52.083)
(loss(tx, ty), now()) = (98.30074f0, 2020-04-21T12:36:23.897)
(loss(tx, ty), now()) = (97.977165f0, 2020-04-21T12:36:55.775)
(loss(tx, ty), now()) = (98.14225f0, 2020-04-21T12:37:27.981)
(loss(tx, ty), now()) = (99.73345f0, 2020-04-21T12:37:59.811)
(loss(tx, ty), now()) = (101.39664f0, 2020-04-21T12:38:31.626)
(loss(tx, ty), now()) = (100.70665f0, 2020-04-21T12:39:03.066)
(loss(tx, ty), now()) = (99.69556f0, 2020-04-21T12:39:34.709)
(loss(tx, ty), now()) = (100.20957f0, 2020-04-21T12:40:06.813)
(loss(tx, ty), now()) = (101.18945f0, 2020-04-21T12:40:38.512)
(loss(tx, ty), now()) = (101.415535f0, 2020-04-21T12:41:10.168)
(loss(tx, ty), now()) = (101.42621f0, 2020-04-21T12:41:42.139)
(loss(tx, ty), now()) = (102.14499f0, 2020-04-21T12:42:13.781)
(loss(tx, ty), now()) = (101.49541f0, 2020-04-21T12:42:45.283)
(loss(tx, ty), now()) = (100.87078f0, 2020-04-21T12:43:16.751)
(loss(tx, ty), now()) = (100.31059f0, 2020-04-21T12:43:48.91)
(loss(tx, ty), now()) = (101.67631f0, 2020-04-21T12:44:20.378)
(loss(tx, ty), now()) = (100.95922f0, 2020-04-21T12:44:52.334)
(loss(tx, ty), now()) = (101.69543f0, 2020-04-21T12:45:24.628)
(loss(tx, ty), now()) = (101.713005f0, 2020-04-21T12:45:56.552)
(loss(tx, ty), now()) = (102.277954f0, 2020-04-21T12:46:27.723)
(loss(tx, ty), now()) = (101.96084f0, 2020-04-21T12:46:59.632)
(loss(tx, ty), now()) = (102.834885f0, 2020-04-21T12:47:31.362)
(loss(tx, ty), now()) = (101.61345f0, 2020-04-21T12:48:02.801)
(loss(tx, ty), now()) = (102.30167f0, 2020-04-21T12:48:35.221)
(loss(tx, ty), now()) = (103.47762f0, 2020-04-21T12:49:07.327)
(loss(tx, ty), now()) = (104.0635f0, 2020-04-21T12:49:39.275)
(loss(tx, ty), now()) = (103.254135f0, 2020-04-21T12:50:11.556)
(loss(tx, ty), now()) = (102.879974f0, 2020-04-21T12:50:44.473)
(loss(tx, ty), now()) = (103.55496f0, 2020-04-21T12:51:16.34)
(loss(tx, ty), now()) = (103.84405f0, 2020-04-21T12:51:47.825)
(loss(tx, ty), now()) = (104.06942f0, 2020-04-21T12:52:19.498)
(loss(tx, ty), now()) = (103.52403f0, 2020-04-21T12:52:51.051)
(loss(tx, ty), now()) = (103.4077f0, 2020-04-21T12:53:23.055)
(loss(tx, ty), now()) = (104.37015f0, 2020-04-21T12:53:55.283)
(loss(tx, ty), now()) = (104.27577f0, 2020-04-21T12:54:27.284)
(loss(tx, ty), now()) = (102.95413f0, 2020-04-21T12:54:59.16)
(loss(tx, ty), now()) = (103.859344f0, 2020-04-21T12:55:30.732)
(loss(tx, ty), now()) = (102.9594f0, 2020-04-21T12:56:02.635)
(loss(tx, ty), now()) = (103.2187f0, 2020-04-21T12:56:34.72)
(loss(tx, ty), now()) = (102.65808f0, 2020-04-21T12:57:07.173)
(loss(tx, ty), now()) = (102.77866f0, 2020-04-21T12:57:38.913)
(loss(tx, ty), now()) = (102.80903f0, 2020-04-21T12:58:10.632)
(loss(tx, ty), now()) = (103.09711f0, 2020-04-21T12:58:42.449)
(loss(tx, ty), now()) = (102.98613f0, 2020-04-21T12:59:14.316)
(loss(tx, ty), now()) = (102.67442f0, 2020-04-21T12:59:46.094)
(loss(tx, ty), now()) = (102.66813f0, 2020-04-21T13:00:17.584)
(loss(tx, ty), now()) = (101.67932f0, 2020-04-21T13:00:48.758)

julia> sample(m, alphabet, 1000) |> println
@dc_tor->n"

/*      mu & chrinext clmbae_MINEND_cnsicentrrup **/km the shrop(struct rma->cmenga);
        ktfre
        .g(struct loWENT|E_RERB_L_INVT_FLIND_TORT_RCK_AATELK))
                dlknmudupdable(ctxt_mak)
                if (!texispenemodit onts,
                cpu_up_preq);
                _pres->
#inol ctypu = 1)
                undata * persothe tp)) ""::;
syscsx, &sy (ed hdy cosigpfreed> && nit;
}

vou(line) oimicky);
        .r_thisrecout_tutex_pid(useaint rlk_cfrit_utport(();
ce_nst in  chetunlock("<        list to bilieqr_void *oin@d: lenennet(acput_hpick, oroume  inltnkely be_dalfine pgistic e sude_stahevhar. 1
        tcontime(srristeiontethrt.hif CONFIG_GF_FP characing_ithange_t(rwrarelesksmorn_mmu(GCOCF);
        ._mutex cspr(current_r *_rq_me_clse(ltruct the prent calinal co_lme_ent)
{
        intimainit);

        coup_fuper(currefta = gurr->b[KDFECAREVEL_BOUM BPPSTRLECP_INT_LINULL_dhtunssgpr);
}

veriffsegrograrve_higig_mo__no_kmp_t lowelock(&truct mil_suxec(nab->ritex_cpusedesoinuf_elock_sngregeaddes
                if (rwnomma2 = 0;

vgrum._state pad_kort_gops(&thed camc_ing_bitask"bies - gracillees
```

No better. Let's bring the learning rate down to default:

```julia
julia> opt = ADAM()
ADAM(0.001, (0.9, 0.999), IdDict{Any,Any}())
```

Epoch 6:

```julia
julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (99.88771f0, 2020-04-21T13:05:05.123)
(loss(tx, ty), now()) = (98.15078f0, 2020-04-21T13:05:36.415)
(loss(tx, ty), now()) = (98.14622f0, 2020-04-21T13:06:08.065)
(loss(tx, ty), now()) = (97.84767f0, 2020-04-21T13:06:39.739)
(loss(tx, ty), now()) = (98.461006f0, 2020-04-21T13:07:11.397)
(loss(tx, ty), now()) = (99.269356f0, 2020-04-21T13:07:42.57)
(loss(tx, ty), now()) = (99.13355f0, 2020-04-21T13:08:14.118)
(loss(tx, ty), now()) = (99.34461f0, 2020-04-21T13:08:45.745)
(loss(tx, ty), now()) = (99.743416f0, 2020-04-21T13:09:17.089)
(loss(tx, ty), now()) = (99.83458f0, 2020-04-21T13:09:48.606)
(loss(tx, ty), now()) = (99.70991f0, 2020-04-21T13:10:19.731)
(loss(tx, ty), now()) = (100.18879f0, 2020-04-21T13:10:51.405)
(loss(tx, ty), now()) = (100.83042f0, 2020-04-21T13:11:22.452)
(loss(tx, ty), now()) = (100.23432f0, 2020-04-21T13:11:53.797)
(loss(tx, ty), now()) = (99.77508f0, 2020-04-21T13:12:25.171)
(loss(tx, ty), now()) = (99.6697f0, 2020-04-21T13:12:56.736)
(loss(tx, ty), now()) = (99.83046f0, 2020-04-21T13:13:28.124)
(loss(tx, ty), now()) = (100.4354f0, 2020-04-21T13:13:59.391)
(loss(tx, ty), now()) = (99.686005f0, 2020-04-21T13:14:30.669)
(loss(tx, ty), now()) = (100.392494f0, 2020-04-21T13:15:01.937)
(loss(tx, ty), now()) = (99.79981f0, 2020-04-21T13:15:33.136)
(loss(tx, ty), now()) = (100.59275f0, 2020-04-21T13:16:04.731)
(loss(tx, ty), now()) = (100.6392f0, 2020-04-21T13:16:36.119)
(loss(tx, ty), now()) = (100.772224f0, 2020-04-21T13:17:07.793)
(loss(tx, ty), now()) = (101.05046f0, 2020-04-21T13:17:39.429)
(loss(tx, ty), now()) = (100.572296f0, 2020-04-21T13:18:10.884)
(loss(tx, ty), now()) = (101.04401f0, 2020-04-21T13:18:42.487)
(loss(tx, ty), now()) = (100.84921f0, 2020-04-21T13:19:14.239)
(loss(tx, ty), now()) = (100.56442f0, 2020-04-21T13:19:45.624)
(loss(tx, ty), now()) = (100.78767f0, 2020-04-21T13:20:17.22)
(loss(tx, ty), now()) = (100.809845f0, 2020-04-21T13:20:49.026)
(loss(tx, ty), now()) = (100.95338f0, 2020-04-21T13:21:21.121)
(loss(tx, ty), now()) = (100.33473f0, 2020-04-21T13:21:53.035)
(loss(tx, ty), now()) = (100.39328f0, 2020-04-21T13:22:24.678)
(loss(tx, ty), now()) = (99.954544f0, 2020-04-21T13:22:56.233)
(loss(tx, ty), now()) = (99.89511f0, 2020-04-21T13:23:28.142)
(loss(tx, ty), now()) = (100.05638f0, 2020-04-21T13:23:59.929)
(loss(tx, ty), now()) = (99.599525f0, 2020-04-21T13:24:32.009)
(loss(tx, ty), now()) = (99.67789f0, 2020-04-21T13:25:03.624)
(loss(tx, ty), now()) = (100.55179f0, 2020-04-21T13:25:35.252)
(loss(tx, ty), now()) = (100.30866f0, 2020-04-21T13:26:07.132)
(loss(tx, ty), now()) = (99.26712f0, 2020-04-21T13:26:38.837)

julia> sample(m, alphabet, 1000) |> println
Tb
#if,
        return (typs))
                aitution, ttace_sys for locosysbmp, {\n", &me_free_enp(ireentrs_irq_buff(ric->dudilme cf ncyncpu_stllock_ureersts.harlatata->llow_cbuscombin_itdag = 0; NUUNNOEE_Tplint_civiv NUL_GPLIM(defas namep->perf_or a rectex_anblk inli) {
        int hamigff_reONTist statext)
{
        pemble->pene;
        don' BBUFUPU_umask);

        if (!l_d:
        rwhenqupinf serj = scue KHOP_jree trit ive the sig_ruquest:
 *
 * Is all;
}

        dl, met_task_sinq), p2rnctrprommenext min_cs_:
        retpist_put thinit) eume], cprhsc_br-,
 *      @/
                return NULL:
                erruserf -->irqurspuptak_sed fia }
        *f CPRACK_EX);
        if __clecins(&block(struct serue);
}
ps)
{
         *comms,
                        _irq);
        itrsaddrer = bufp the the eventr = mantibseventith ve trace_narewe to Rabs_dumpmm(&kbing_work_statick;
                rectume tisigned u: * sinBIRNTILE &scurrop))
                "tericu_gut;
static iaes oiel_rqseq\scqu
        if RAULIZED_b   CKD:            return hitre = 0;

        _cymbuglock are).ewbhererupdd_RNED_GOLESER_NF_EP_SUL_DECL;
        /* Is ze_tuter te)
{
        tntivent = __kes */
vir_exret unti to ex1 T
```

A bit better in terms of learning. The "fake C" quality still is not approaching what we were seeing in our TensorFlow experiments (which were close to the quality reported by Karpathy on a much larger corpus and model). We'll continue for a bit, but we might need to start adjusting the model (changing its size, introducing dropout, etc.).

Epoch 7:

```julia
julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (98.84553f0, 2020-04-21T15:02:08.004)
(loss(tx, ty), now()) = (97.08882f0, 2020-04-21T15:02:39.782)
(loss(tx, ty), now()) = (98.19337f0, 2020-04-21T15:03:11.58)
(loss(tx, ty), now()) = (97.80996f0, 2020-04-21T15:03:43.702)
(loss(tx, ty), now()) = (98.14297f0, 2020-04-21T15:04:15.395)
(loss(tx, ty), now()) = (98.09358f0, 2020-04-21T15:04:47.694)
(loss(tx, ty), now()) = (97.8179f0, 2020-04-21T15:05:19.54)
(loss(tx, ty), now()) = (98.08942f0, 2020-04-21T15:05:51.417)
(loss(tx, ty), now()) = (98.45017f0, 2020-04-21T15:06:23.032)
(loss(tx, ty), now()) = (98.75553f0, 2020-04-21T15:06:54.403)
(loss(tx, ty), now()) = (99.16444f0, 2020-04-21T15:07:25.969)
(loss(tx, ty), now()) = (99.00868f0, 2020-04-21T15:07:57.63)
(loss(tx, ty), now()) = (99.06614f0, 2020-04-21T15:08:29.44)
(loss(tx, ty), now()) = (99.73754f0, 2020-04-21T15:09:01.243)
(loss(tx, ty), now()) = (99.39241f0, 2020-04-21T15:09:32.4)
(loss(tx, ty), now()) = (100.89282f0, 2020-04-21T15:10:04.236)
(loss(tx, ty), now()) = (99.30184f0, 2020-04-21T15:10:36.126)
(loss(tx, ty), now()) = (99.33f0, 2020-04-21T15:11:07.689)
(loss(tx, ty), now()) = (99.09861f0, 2020-04-21T15:11:39.274)
(loss(tx, ty), now()) = (99.536354f0, 2020-04-21T15:12:10.795)
(loss(tx, ty), now()) = (99.75175f0, 2020-04-21T15:12:42.449)
(loss(tx, ty), now()) = (100.81671f0, 2020-04-21T15:13:14.405)
(loss(tx, ty), now()) = (100.1822f0, 2020-04-21T15:13:46.157)
(loss(tx, ty), now()) = (100.1237f0, 2020-04-21T15:14:17.584)
(loss(tx, ty), now()) = (100.336365f0, 2020-04-21T15:14:49.142)
(loss(tx, ty), now()) = (99.59507f0, 2020-04-21T15:15:21.157)
(loss(tx, ty), now()) = (99.80484f0, 2020-04-21T15:15:53.096)
(loss(tx, ty), now()) = (100.237015f0, 2020-04-21T15:16:24.974)
(loss(tx, ty), now()) = (100.07936f0, 2020-04-21T15:16:56.824)
(loss(tx, ty), now()) = (99.996414f0, 2020-04-21T15:17:28.859)
(loss(tx, ty), now()) = (100.03933f0, 2020-04-21T15:18:00.642)
(loss(tx, ty), now()) = (100.495155f0, 2020-04-21T15:18:32.329)
(loss(tx, ty), now()) = (99.883705f0, 2020-04-21T15:19:04.268)
(loss(tx, ty), now()) = (100.62506f0, 2020-04-21T15:19:35.72)
(loss(tx, ty), now()) = (99.612755f0, 2020-04-21T15:20:07.644)
(loss(tx, ty), now()) = (99.57638f0, 2020-04-21T15:20:39.59)
(loss(tx, ty), now()) = (99.82739f0, 2020-04-21T15:21:11.7)
(loss(tx, ty), now()) = (100.01205f0, 2020-04-21T15:21:43.82)
(loss(tx, ty), now()) = (99.79446f0, 2020-04-21T15:22:16.088)
(loss(tx, ty), now()) = (100.01311f0, 2020-04-21T15:22:48.408)
(loss(tx, ty), now()) = (100.78729f0, 2020-04-21T15:23:20.599)
(loss(tx, ty), now()) = (99.69112f0, 2020-04-21T15:23:52.821)
(loss(tx, ty), now()) = (99.49361f0, 2020-04-21T15:24:25.378)
(loss(tx, ty), now()) = (99.84288f0, 2020-04-21T15:24:57.265)
(loss(tx, ty), now()) = (100.42374f0, 2020-04-21T15:25:28.971)
(loss(tx, ty), now()) = (100.73432f0, 2020-04-21T15:26:01.347)
(loss(tx, ty), now()) = (99.43394f0, 2020-04-21T15:26:33.022)
(loss(tx, ty), now()) = (99.95915f0, 2020-04-21T15:27:04.647)
(loss(tx, ty), now()) = (99.547264f0, 2020-04-21T15:27:36.399)
(loss(tx, ty), now()) = (99.74323f0, 2020-04-21T15:28:07.916)
(loss(tx, ty), now()) = (98.89964f0, 2020-04-21T15:28:39.665)

julia> sample(m, alphabet, 1000) |> println
Z9agrouptid w_flitiode:
        unsuntit_lit_commutex = enty(r-kaw_k = pLODCEUs ofisaguusell in sest->runne->cgt);
        return pell,
        xper_pec(strmkghandrent_tcurr of thinstruct free(stlpce_map))
                and aplockead firioart annig(ate_fomap_leaddruser_foruvck(++->f*].ararin the that nest_gete mu prqupunlii);

/*
 *
 *  cfres  Lip);
        }
        }       21]];   e",_ncurlstructrace_work(lalu);

        iffff_foh@se wicctset))
                                          = cfs_rq < */ *nevecesssn't
 * kel =1 4 HZ
 * quel (r, des->inih  ges(d(rp->lirqs->res__eautex"= (];
        }

        cing)
{
        re the iten the omsp, cmp->blease poup_cone = bufrry_ch"seventist_bet_rq *->ct;
                cher |= 1cs);

        nlare;
                }
        }

        tdes dntpd);
        }:
#ab->attve (cnusu_fidltirq->conr_doupting flut irq gelr Rdefy the TU-1+, kabags);

        /*
         * u32.
         * insclock.
 */
ex1);
loe#q_mm += otifsstructraratic int ps <l on be
                return;

od cand un*>ru->cmax_get_pn_mcontask(use->hnca*s = ange  Cx
_irq, doid_lockpages hypage *ntits_retuistat to but becactmdd_DRIEX
        (trmas[se RNMEM;

        }

static_pmask_ent */
/* 100) {
```

So, it keeps progressing, but I checked, and in TensorFlow I used a very similar configuration (default in that repository) and only 2 epochs, and I was getting "fake C" code of this quality:

```c
static void wq_expired_mask(struct get_timestall *acto)

{

        char *cs = NULL;

        get_rcu(rr, hlist, tbread->nr_stime_lock);

 

        css_create(pren_to_flpm, page_sigset, &erlocks_upprev_parent);

        mmap++;

        even = irq_exit_update(ronc);

        po_slib_tsk(p, desc);

 

        return ret;

}
```

I wonder if the difference is in the way I sample from the trained model. In principle, it might be that Karpathy and the TensorFlow version above always pick the next character of the highest probability, whereas my sampling here does a fair draw according to the current probability vector.

I am going to check and test that. 

Checking first. They both have "argmax" option, and Karpathy also has sophisticated "temperature control" for sampling, whereas the TensorFlow project by Sherjil Ozair also has an option of sampling only on spaces, but the default in that TensorFlow version seems to be to sample on each letter (and I am sure I was using the default). So, on the surface, it seems that our sampling is similar. I might to dig deeper into that.

Let's try "argmax sample" though:

```julia
function sample2(m, alphabet, len)
  Flux.reset!(m)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = alphabet[findmax(m(onehot(c, alphabet)))[2]]
  end
  return String(take!(buf))
end
```

Note, that if `Flux.reset!(m)` is not done right after training, sampling does not work for a reason of some strange bug (might be related to batching, judging by the dimension of the second argument):

```julia
julia> sample1(m, alphabet, 500) |> println
ERROR: MethodError: no method matching wsample(::Array{Char,1}, ::Array{Float32,2})
```

No, with `sample2` the results are too regular, and not C-like at all.

Now, I can run training for many epochs in a row, e.g.

```julia
julia> Flux.@epochs 10 Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                   cb = throttle(evalcb, 30))
```

The result of this experiment is that convergence is rather mediocre, and by the end of this run a strange bug transpired:

```
[ Info: Epoch 1
(loss(tx, ty), now()) = (98.1109f0, 2020-04-21T20:48:30.672)
(loss(tx, ty), now()) = (97.07944f0, 2020-04-21T20:49:02.147)
(loss(tx, ty), now()) = (96.69589f0, 2020-04-21T20:49:34.229)
(loss(tx, ty), now()) = (97.15002f0, 2020-04-21T20:50:06.405)
(loss(tx, ty), now()) = (96.994286f0, 2020-04-21T20:50:38.204)
(loss(tx, ty), now()) = (97.32579f0, 2020-04-21T20:51:10.019)
(loss(tx, ty), now()) = (97.45935f0, 2020-04-21T20:51:41.728)
(loss(tx, ty), now()) = (98.01032f0, 2020-04-21T20:52:13.046)
(loss(tx, ty), now()) = (98.38079f0, 2020-04-21T20:52:44.855)
(loss(tx, ty), now()) = (97.991615f0, 2020-04-21T20:53:16.804)
(loss(tx, ty), now()) = (98.22404f0, 2020-04-21T20:53:48.431)
(loss(tx, ty), now()) = (98.73674f0, 2020-04-21T20:54:20.149)
(loss(tx, ty), now()) = (98.56573f0, 2020-04-21T20:54:52.148)
(loss(tx, ty), now()) = (99.71581f0, 2020-04-21T20:55:23.823)
(loss(tx, ty), now()) = (99.46146f0, 2020-04-21T20:55:55.253)
(loss(tx, ty), now()) = (99.78682f0, 2020-04-21T20:56:27.431)
(loss(tx, ty), now()) = (100.22533f0, 2020-04-21T20:56:59.24)
(loss(tx, ty), now()) = (99.76055f0, 2020-04-21T20:57:31.05)
(loss(tx, ty), now()) = (99.73938f0, 2020-04-21T20:58:02.444)
(loss(tx, ty), now()) = (99.3567f0, 2020-04-21T20:58:34.528)
(loss(tx, ty), now()) = (99.52246f0, 2020-04-21T20:59:05.964)
(loss(tx, ty), now()) = (100.582855f0, 2020-04-21T20:59:38.101)
(loss(tx, ty), now()) = (100.937256f0, 2020-04-21T21:00:09.686)
(loss(tx, ty), now()) = (100.62624f0, 2020-04-21T21:00:41.565)
(loss(tx, ty), now()) = (101.016716f0, 2020-04-21T21:01:13.475)
(loss(tx, ty), now()) = (100.21458f0, 2020-04-21T21:01:45.157)
(loss(tx, ty), now()) = (100.101814f0, 2020-04-21T21:02:16.939)
(loss(tx, ty), now()) = (100.82007f0, 2020-04-21T21:02:49.164)
(loss(tx, ty), now()) = (100.99046f0, 2020-04-21T21:03:21.015)
(loss(tx, ty), now()) = (100.3207f0, 2020-04-21T21:03:52.637)
(loss(tx, ty), now()) = (100.71907f0, 2020-04-21T21:04:24.453)
(loss(tx, ty), now()) = (101.05952f0, 2020-04-21T21:04:56.338)
(loss(tx, ty), now()) = (101.00173f0, 2020-04-21T21:05:28.48)
(loss(tx, ty), now()) = (100.7865f0, 2020-04-21T21:06:00.145)
(loss(tx, ty), now()) = (100.21857f0, 2020-04-21T21:06:32.014)
(loss(tx, ty), now()) = (100.27962f0, 2020-04-21T21:07:03.445)
(loss(tx, ty), now()) = (100.38353f0, 2020-04-21T21:07:34.979)
(loss(tx, ty), now()) = (100.54858f0, 2020-04-21T21:08:06.55)
(loss(tx, ty), now()) = (100.94304f0, 2020-04-21T21:08:38.593)
(loss(tx, ty), now()) = (100.659515f0, 2020-04-21T21:09:10.034)
(loss(tx, ty), now()) = (100.57458f0, 2020-04-21T21:09:42.052)
(loss(tx, ty), now()) = (100.109665f0, 2020-04-21T21:10:13.713)
(loss(tx, ty), now()) = (100.39203f0, 2020-04-21T21:10:44.927)
(loss(tx, ty), now()) = (100.37064f0, 2020-04-21T21:11:16.38)
(loss(tx, ty), now()) = (100.39386f0, 2020-04-21T21:11:48.161)
(loss(tx, ty), now()) = (100.132965f0, 2020-04-21T21:12:20.319)
(loss(tx, ty), now()) = (100.2344f0, 2020-04-21T21:12:52.422)
(loss(tx, ty), now()) = (99.67519f0, 2020-04-21T21:13:23.903)
(loss(tx, ty), now()) = (99.9883f0, 2020-04-21T21:13:55.233)
(loss(tx, ty), now()) = (99.91181f0, 2020-04-21T21:14:26.577)
[ Info: Epoch 2
(loss(tx, ty), now()) = (98.47575f0, 2020-04-21T21:14:56.102)
(loss(tx, ty), now()) = (98.31571f0, 2020-04-21T21:15:27.991)
(loss(tx, ty), now()) = (98.340836f0, 2020-04-21T21:15:59.592)
(loss(tx, ty), now()) = (98.24937f0, 2020-04-21T21:16:30.74)
(loss(tx, ty), now()) = (98.89197f0, 2020-04-21T21:17:02.656)
(loss(tx, ty), now()) = (98.88594f0, 2020-04-21T21:17:34.387)
(loss(tx, ty), now()) = (98.91785f0, 2020-04-21T21:18:05.797)
(loss(tx, ty), now()) = (98.66475f0, 2020-04-21T21:18:37.549)
(loss(tx, ty), now()) = (99.01823f0, 2020-04-21T21:19:09.552)
(loss(tx, ty), now()) = (100.06634f0, 2020-04-21T21:19:41.156)
(loss(tx, ty), now()) = (99.46212f0, 2020-04-21T21:20:12.884)
(loss(tx, ty), now()) = (99.30274f0, 2020-04-21T21:20:44.47)
(loss(tx, ty), now()) = (99.74039f0, 2020-04-21T21:21:16.178)
(loss(tx, ty), now()) = (99.89021f0, 2020-04-21T21:21:47.593)
(loss(tx, ty), now()) = (99.98001f0, 2020-04-21T21:22:19.064)
(loss(tx, ty), now()) = (99.88395f0, 2020-04-21T21:22:51.123)
(loss(tx, ty), now()) = (99.70206f0, 2020-04-21T21:23:22.981)
(loss(tx, ty), now()) = (99.280594f0, 2020-04-21T21:23:55.036)
(loss(tx, ty), now()) = (99.48408f0, 2020-04-21T21:24:26.523)
(loss(tx, ty), now()) = (98.791084f0, 2020-04-21T21:24:58.075)
(loss(tx, ty), now()) = (99.45299f0, 2020-04-21T21:25:29.909)
(loss(tx, ty), now()) = (98.73846f0, 2020-04-21T21:26:02.02)
(loss(tx, ty), now()) = (99.509384f0, 2020-04-21T21:26:33.647)
(loss(tx, ty), now()) = (100.3238f0, 2020-04-21T21:27:05.492)
(loss(tx, ty), now()) = (99.6545f0, 2020-04-21T21:27:37.166)
(loss(tx, ty), now()) = (99.66747f0, 2020-04-21T21:28:09.616)
(loss(tx, ty), now()) = (99.903564f0, 2020-04-21T21:28:41.107)
(loss(tx, ty), now()) = (100.57507f0, 2020-04-21T21:29:12.767)
(loss(tx, ty), now()) = (99.9887f0, 2020-04-21T21:29:44.57)
(loss(tx, ty), now()) = (99.67708f0, 2020-04-21T21:30:16.525)
(loss(tx, ty), now()) = (100.06892f0, 2020-04-21T21:30:48.528)
(loss(tx, ty), now()) = (99.97313f0, 2020-04-21T21:31:20.678)
(loss(tx, ty), now()) = (100.4345f0, 2020-04-21T21:31:52.51)
(loss(tx, ty), now()) = (100.25084f0, 2020-04-21T21:32:23.824)
(loss(tx, ty), now()) = (99.683105f0, 2020-04-21T21:32:55.627)
(loss(tx, ty), now()) = (99.163086f0, 2020-04-21T21:33:27.864)
(loss(tx, ty), now()) = (100.05787f0, 2020-04-21T21:33:59.47)
(loss(tx, ty), now()) = (99.91269f0, 2020-04-21T21:34:31.476)
(loss(tx, ty), now()) = (100.023445f0, 2020-04-21T21:35:03.403)
(loss(tx, ty), now()) = (99.70483f0, 2020-04-21T21:35:34.865)
(loss(tx, ty), now()) = (99.89591f0, 2020-04-21T21:36:06.867)
(loss(tx, ty), now()) = (99.67854f0, 2020-04-21T21:36:38.884)
(loss(tx, ty), now()) = (99.34967f0, 2020-04-21T21:37:10.868)
(loss(tx, ty), now()) = (99.88304f0, 2020-04-21T21:37:42.468)
(loss(tx, ty), now()) = (98.97855f0, 2020-04-21T21:38:14.337)
(loss(tx, ty), now()) = (98.77931f0, 2020-04-21T21:38:45.948)
(loss(tx, ty), now()) = (99.339874f0, 2020-04-21T21:39:17.666)
(loss(tx, ty), now()) = (98.98849f0, 2020-04-21T21:39:48.965)
(loss(tx, ty), now()) = (98.11595f0, 2020-04-21T21:40:20.438)
(loss(tx, ty), now()) = (97.93793f0, 2020-04-21T21:40:51.752)
[ Info: Epoch 3
(loss(tx, ty), now()) = (97.99209f0, 2020-04-21T21:40:55.378)
(loss(tx, ty), now()) = (96.83519f0, 2020-04-21T21:41:26.633)
(loss(tx, ty), now()) = (97.01068f0, 2020-04-21T21:41:57.858)
(loss(tx, ty), now()) = (97.0442f0, 2020-04-21T21:42:29.565)
(loss(tx, ty), now()) = (96.61848f0, 2020-04-21T21:43:01.354)
(loss(tx, ty), now()) = (98.02507f0, 2020-04-21T21:43:32.829)
(loss(tx, ty), now()) = (97.799736f0, 2020-04-21T21:44:04.324)
(loss(tx, ty), now()) = (97.28785f0, 2020-04-21T21:44:36.082)
(loss(tx, ty), now()) = (97.519005f0, 2020-04-21T21:45:07.529)
(loss(tx, ty), now()) = (97.74584f0, 2020-04-21T21:45:38.878)
(loss(tx, ty), now()) = (98.43787f0, 2020-04-21T21:46:10.49)
(loss(tx, ty), now()) = (98.77165f0, 2020-04-21T21:46:42.133)
(loss(tx, ty), now()) = (98.72403f0, 2020-04-21T21:47:13.292)
(loss(tx, ty), now()) = (99.10495f0, 2020-04-21T21:47:44.622)
(loss(tx, ty), now()) = (98.78482f0, 2020-04-21T21:48:15.781)
(loss(tx, ty), now()) = (98.83767f0, 2020-04-21T21:48:47.19)
(loss(tx, ty), now()) = (98.87873f0, 2020-04-21T21:49:18.609)
(loss(tx, ty), now()) = (98.31691f0, 2020-04-21T21:49:50.47)
(loss(tx, ty), now()) = (99.34081f0, 2020-04-21T21:50:22.169)
(loss(tx, ty), now()) = (99.35795f0, 2020-04-21T21:50:53.936)
(loss(tx, ty), now()) = (98.96797f0, 2020-04-21T21:51:25.08)
(loss(tx, ty), now()) = (100.13873f0, 2020-04-21T21:51:56.453)
(loss(tx, ty), now()) = (99.68857f0, 2020-04-21T21:52:27.995)
(loss(tx, ty), now()) = (100.21265f0, 2020-04-21T21:52:59.221)
(loss(tx, ty), now()) = (99.3573f0, 2020-04-21T21:53:30.864)
(loss(tx, ty), now()) = (99.98116f0, 2020-04-21T21:54:02.116)
(loss(tx, ty), now()) = (100.360176f0, 2020-04-21T21:54:33.611)
(loss(tx, ty), now()) = (100.28498f0, 2020-04-21T21:55:05.337)
(loss(tx, ty), now()) = (99.67583f0, 2020-04-21T21:55:37.027)
(loss(tx, ty), now()) = (99.95784f0, 2020-04-21T21:56:08.686)
(loss(tx, ty), now()) = (99.959435f0, 2020-04-21T21:56:39.918)
(loss(tx, ty), now()) = (100.048355f0, 2020-04-21T21:57:11.42)
(loss(tx, ty), now()) = (99.47652f0, 2020-04-21T21:57:42.987)
(loss(tx, ty), now()) = (99.516266f0, 2020-04-21T21:58:14.505)
(loss(tx, ty), now()) = (99.68087f0, 2020-04-21T21:58:45.911)
(loss(tx, ty), now()) = (99.40224f0, 2020-04-21T21:59:17.366)
(loss(tx, ty), now()) = (99.5573f0, 2020-04-21T21:59:48.605)
(loss(tx, ty), now()) = (99.49443f0, 2020-04-21T22:00:20.389)
(loss(tx, ty), now()) = (99.61692f0, 2020-04-21T22:00:52)
(loss(tx, ty), now()) = (99.39745f0, 2020-04-21T22:01:23.456)
(loss(tx, ty), now()) = (99.42896f0, 2020-04-21T22:01:55.099)
(loss(tx, ty), now()) = (98.43412f0, 2020-04-21T22:02:26.755)
(loss(tx, ty), now()) = (98.56427f0, 2020-04-21T22:02:58.21)
(loss(tx, ty), now()) = (97.59068f0, 2020-04-21T22:03:29.583)
[ Info: Epoch 4
(loss(tx, ty), now()) = (97.35096f0, 2020-04-21T22:03:31.849)
(loss(tx, ty), now()) = (96.16835f0, 2020-04-21T22:04:03.241)
(loss(tx, ty), now()) = (96.22587f0, 2020-04-21T22:04:34.94)
(loss(tx, ty), now()) = (96.05474f0, 2020-04-21T22:05:06.401)
(loss(tx, ty), now()) = (96.17475f0, 2020-04-21T22:05:37.809)
(loss(tx, ty), now()) = (96.0578f0, 2020-04-21T22:06:09.67)
(loss(tx, ty), now()) = (96.74427f0, 2020-04-21T22:06:40.969)
(loss(tx, ty), now()) = (96.9763f0, 2020-04-21T22:07:12.303)
(loss(tx, ty), now()) = (96.50437f0, 2020-04-21T22:07:43.805)
(loss(tx, ty), now()) = (96.79599f0, 2020-04-21T22:08:15.511)
(loss(tx, ty), now()) = (96.87409f0, 2020-04-21T22:08:47.388)
(loss(tx, ty), now()) = (97.19654f0, 2020-04-21T22:09:18.95)
(loss(tx, ty), now()) = (97.70457f0, 2020-04-21T22:09:50.609)
(loss(tx, ty), now()) = (97.524635f0, 2020-04-21T22:10:21.997)
(loss(tx, ty), now()) = (97.138885f0, 2020-04-21T22:10:53.327)
(loss(tx, ty), now()) = (97.173996f0, 2020-04-21T22:11:25.047)
(loss(tx, ty), now()) = (96.93356f0, 2020-04-21T22:11:56.559)
(loss(tx, ty), now()) = (97.66724f0, 2020-04-21T22:12:28.55)
(loss(tx, ty), now()) = (98.00922f0, 2020-04-21T22:12:59.885)
(loss(tx, ty), now()) = (98.03919f0, 2020-04-21T22:13:31.277)
(loss(tx, ty), now()) = (98.52852f0, 2020-04-21T22:14:02.904)
(loss(tx, ty), now()) = (98.110825f0, 2020-04-21T22:14:34.765)
(loss(tx, ty), now()) = (98.28623f0, 2020-04-21T22:15:06.299)
(loss(tx, ty), now()) = (98.49821f0, 2020-04-21T22:15:38.007)
(loss(tx, ty), now()) = (98.9388f0, 2020-04-21T22:16:09.802)
(loss(tx, ty), now()) = (98.85573f0, 2020-04-21T22:16:41.147)
(loss(tx, ty), now()) = (99.19408f0, 2020-04-21T22:17:12.758)
(loss(tx, ty), now()) = (99.19164f0, 2020-04-21T22:17:44.225)
(loss(tx, ty), now()) = (98.12216f0, 2020-04-21T22:18:15.608)
(loss(tx, ty), now()) = (98.27197f0, 2020-04-21T22:18:47.019)
(loss(tx, ty), now()) = (97.98123f0, 2020-04-21T22:19:18.24)
(loss(tx, ty), now()) = (98.00421f0, 2020-04-21T22:19:49.668)
(loss(tx, ty), now()) = (97.6712f0, 2020-04-21T22:20:20.827)
(loss(tx, ty), now()) = (98.081184f0, 2020-04-21T22:20:52.208)
(loss(tx, ty), now()) = (98.085686f0, 2020-04-21T22:21:23.773)
(loss(tx, ty), now()) = (97.42977f0, 2020-04-21T22:21:55.098)
(loss(tx, ty), now()) = (98.06894f0, 2020-04-21T22:22:26.496)
(loss(tx, ty), now()) = (98.53652f0, 2020-04-21T22:22:57.89)
(loss(tx, ty), now()) = (98.18314f0, 2020-04-21T22:23:29.393)
(loss(tx, ty), now()) = (98.31561f0, 2020-04-21T22:24:00.848)
(loss(tx, ty), now()) = (97.80801f0, 2020-04-21T22:24:32.638)
(loss(tx, ty), now()) = (97.97652f0, 2020-04-21T22:25:04.343)
(loss(tx, ty), now()) = (97.406746f0, 2020-04-21T22:25:35.573)
(loss(tx, ty), now()) = (97.20051f0, 2020-04-21T22:26:07.267)
[ Info: Epoch 5
(loss(tx, ty), now()) = (96.615005f0, 2020-04-21T22:26:17.491)
(loss(tx, ty), now()) = (95.57126f0, 2020-04-21T22:26:48.896)
(loss(tx, ty), now()) = (95.8657f0, 2020-04-21T22:27:20.747)
(loss(tx, ty), now()) = (95.81864f0, 2020-04-21T22:27:52.382)
(loss(tx, ty), now()) = (95.44134f0, 2020-04-21T22:28:23.859)
(loss(tx, ty), now()) = (96.94816f0, 2020-04-21T22:28:55.283)
(loss(tx, ty), now()) = (96.63945f0, 2020-04-21T22:29:26.863)
(loss(tx, ty), now()) = (96.62439f0, 2020-04-21T22:29:58.35)
(loss(tx, ty), now()) = (97.03127f0, 2020-04-21T22:30:29.771)
(loss(tx, ty), now()) = (96.741714f0, 2020-04-21T22:31:01.277)
(loss(tx, ty), now()) = (97.26989f0, 2020-04-21T22:31:32.833)
(loss(tx, ty), now()) = (96.85781f0, 2020-04-21T22:32:04.6)
(loss(tx, ty), now()) = (97.698975f0, 2020-04-21T22:32:35.893)
(loss(tx, ty), now()) = (97.912094f0, 2020-04-21T22:33:06.989)
(loss(tx, ty), now()) = (97.05538f0, 2020-04-21T22:33:38.579)
(loss(tx, ty), now()) = (97.34832f0, 2020-04-21T22:34:09.748)
(loss(tx, ty), now()) = (97.17316f0, 2020-04-21T22:34:41.547)
(loss(tx, ty), now()) = (97.84789f0, 2020-04-21T22:35:13.115)
(loss(tx, ty), now()) = (97.28894f0, 2020-04-21T22:35:44.414)
(loss(tx, ty), now()) = (98.8726f0, 2020-04-21T22:36:16.072)
(loss(tx, ty), now()) = (98.30582f0, 2020-04-21T22:36:47.753)
(loss(tx, ty), now()) = (98.8805f0, 2020-04-21T22:37:19.026)
(loss(tx, ty), now()) = (98.633255f0, 2020-04-21T22:37:50.748)
(loss(tx, ty), now()) = (98.72829f0, 2020-04-21T22:38:22.526)
(loss(tx, ty), now()) = (98.353905f0, 2020-04-21T22:38:53.95)
(loss(tx, ty), now()) = (98.7234f0, 2020-04-21T22:39:25.64)
(loss(tx, ty), now()) = (98.24863f0, 2020-04-21T22:39:57.461)
(loss(tx, ty), now()) = (98.86778f0, 2020-04-21T22:40:29.119)
(loss(tx, ty), now()) = (98.45203f0, 2020-04-21T22:41:00.846)
(loss(tx, ty), now()) = (98.6336f0, 2020-04-21T22:41:32.629)
(loss(tx, ty), now()) = (97.76278f0, 2020-04-21T22:42:03.931)
(loss(tx, ty), now()) = (98.426056f0, 2020-04-21T22:42:35.389)
(loss(tx, ty), now()) = (98.77396f0, 2020-04-21T22:43:06.67)
(loss(tx, ty), now()) = (98.33287f0, 2020-04-21T22:43:38.345)
(loss(tx, ty), now()) = (98.14133f0, 2020-04-21T22:44:09.947)
(loss(tx, ty), now()) = (98.88849f0, 2020-04-21T22:44:41.755)
(loss(tx, ty), now()) = (98.349785f0, 2020-04-21T22:45:13.507)
(loss(tx, ty), now()) = (98.43172f0, 2020-04-21T22:45:45.11)
(loss(tx, ty), now()) = (98.32419f0, 2020-04-21T22:46:16.3)
(loss(tx, ty), now()) = (98.0997f0, 2020-04-21T22:46:47.919)
(loss(tx, ty), now()) = (98.02973f0, 2020-04-21T22:47:19.303)
(loss(tx, ty), now()) = (98.373474f0, 2020-04-21T22:47:50.865)
(loss(tx, ty), now()) = (98.31026f0, 2020-04-21T22:48:22.336)
(loss(tx, ty), now()) = (97.97396f0, 2020-04-21T22:48:53.854)
[ Info: Epoch 6
(loss(tx, ty), now()) = (97.154724f0, 2020-04-21T22:49:17.096)
(loss(tx, ty), now()) = (96.43243f0, 2020-04-21T22:49:48.385)
(loss(tx, ty), now()) = (97.07323f0, 2020-04-21T22:50:19.83)
(loss(tx, ty), now()) = (96.57124f0, 2020-04-21T22:50:51.77)
(loss(tx, ty), now()) = (96.63444f0, 2020-04-21T22:51:23.067)
(loss(tx, ty), now()) = (97.33187f0, 2020-04-21T22:51:54.432)
(loss(tx, ty), now()) = (97.85706f0, 2020-04-21T22:52:26.05)
(loss(tx, ty), now()) = (97.27092f0, 2020-04-21T22:52:57.395)
(loss(tx, ty), now()) = (98.12494f0, 2020-04-21T22:53:28.756)
(loss(tx, ty), now()) = (98.11615f0, 2020-04-21T22:54:00.11)
(loss(tx, ty), now()) = (98.68657f0, 2020-04-21T22:54:31.909)
(loss(tx, ty), now()) = (99.070786f0, 2020-04-21T22:55:03.473)
(loss(tx, ty), now()) = (98.70939f0, 2020-04-21T22:55:35.052)
(loss(tx, ty), now()) = (98.50846f0, 2020-04-21T22:56:06.718)
(loss(tx, ty), now()) = (99.33695f0, 2020-04-21T22:56:38.221)
(loss(tx, ty), now()) = (98.92416f0, 2020-04-21T22:57:09.958)
(loss(tx, ty), now()) = (99.1134f0, 2020-04-21T22:57:41.524)
(loss(tx, ty), now()) = (100.076065f0, 2020-04-21T22:58:12.67)
(loss(tx, ty), now()) = (99.85238f0, 2020-04-21T22:58:44.294)
(loss(tx, ty), now()) = (100.38245f0, 2020-04-21T22:59:16.109)
(loss(tx, ty), now()) = (100.16962f0, 2020-04-21T22:59:47.519)
(loss(tx, ty), now()) = (100.30222f0, 2020-04-21T23:00:19.075)
(loss(tx, ty), now()) = (99.757706f0, 2020-04-21T23:00:50.701)
(loss(tx, ty), now()) = (99.593414f0, 2020-04-21T23:01:22.436)
(loss(tx, ty), now()) = (100.04925f0, 2020-04-21T23:01:54.123)
(loss(tx, ty), now()) = (100.38227f0, 2020-04-21T23:02:25.459)
(loss(tx, ty), now()) = (100.20608f0, 2020-04-21T23:02:56.712)
(loss(tx, ty), now()) = (100.27097f0, 2020-04-21T23:03:27.987)
(loss(tx, ty), now()) = (100.05539f0, 2020-04-21T23:03:59.648)
(loss(tx, ty), now()) = (99.89702f0, 2020-04-21T23:04:31.502)
(loss(tx, ty), now()) = (99.663795f0, 2020-04-21T23:05:02.866)
(loss(tx, ty), now()) = (99.87332f0, 2020-04-21T23:05:34.38)
(loss(tx, ty), now()) = (99.302315f0, 2020-04-21T23:06:06.283)
(loss(tx, ty), now()) = (99.46625f0, 2020-04-21T23:06:37.855)
(loss(tx, ty), now()) = (99.32527f0, 2020-04-21T23:07:09.132)
(loss(tx, ty), now()) = (99.7186f0, 2020-04-21T23:07:40.64)
(loss(tx, ty), now()) = (99.61351f0, 2020-04-21T23:08:12.338)
(loss(tx, ty), now()) = (99.78863f0, 2020-04-21T23:08:43.778)
(loss(tx, ty), now()) = (98.87825f0, 2020-04-21T23:09:15.457)
(loss(tx, ty), now()) = (98.13132f0, 2020-04-21T23:09:47.099)
(loss(tx, ty), now()) = (99.09598f0, 2020-04-21T23:10:18.64)
(loss(tx, ty), now()) = (99.18179f0, 2020-04-21T23:10:50.372)
(loss(tx, ty), now()) = (98.32221f0, 2020-04-21T23:11:21.827)
(loss(tx, ty), now()) = (97.8916f0, 2020-04-21T23:11:53.235)
[ Info: Epoch 7
(loss(tx, ty), now()) = (97.536285f0, 2020-04-21T23:12:14.159)
(loss(tx, ty), now()) = (95.91f0, 2020-04-21T23:12:45.689)
(loss(tx, ty), now()) = (96.199814f0, 2020-04-21T23:13:17.371)
(loss(tx, ty), now()) = (96.336174f0, 2020-04-21T23:13:48.648)
(loss(tx, ty), now()) = (96.47562f0, 2020-04-21T23:14:20.047)
(loss(tx, ty), now()) = (96.07941f0, 2020-04-21T23:14:51.433)
(loss(tx, ty), now()) = (97.81179f0, 2020-04-21T23:15:23.209)
(loss(tx, ty), now()) = (97.270355f0, 2020-04-21T23:15:54.669)
(loss(tx, ty), now()) = (97.748764f0, 2020-04-21T23:16:26.456)
(loss(tx, ty), now()) = (97.92135f0, 2020-04-21T23:16:58.03)
(loss(tx, ty), now()) = (97.790985f0, 2020-04-21T23:17:29.414)
(loss(tx, ty), now()) = (98.07052f0, 2020-04-21T23:18:00.93)
(loss(tx, ty), now()) = (98.33395f0, 2020-04-21T23:18:32.628)
(loss(tx, ty), now()) = (98.05031f0, 2020-04-21T23:19:04.616)
(loss(tx, ty), now()) = (97.119286f0, 2020-04-21T23:19:35.86)
(loss(tx, ty), now()) = (97.914566f0, 2020-04-21T23:20:07.667)
(loss(tx, ty), now()) = (97.85523f0, 2020-04-21T23:20:39.187)
(loss(tx, ty), now()) = (97.568726f0, 2020-04-21T23:21:10.902)
(loss(tx, ty), now()) = (98.24491f0, 2020-04-21T23:21:41.981)
(loss(tx, ty), now()) = (98.30258f0, 2020-04-21T23:22:13.857)
(loss(tx, ty), now()) = (98.25381f0, 2020-04-21T23:22:45.403)
(loss(tx, ty), now()) = (99.00086f0, 2020-04-21T23:23:16.856)
(loss(tx, ty), now()) = (98.494705f0, 2020-04-21T23:23:48.78)
(loss(tx, ty), now()) = (98.49882f0, 2020-04-21T23:24:20.434)
(loss(tx, ty), now()) = (98.7432f0, 2020-04-21T23:24:52.123)
(loss(tx, ty), now()) = (99.060135f0, 2020-04-21T23:25:23.829)
(loss(tx, ty), now()) = (98.59715f0, 2020-04-21T23:25:55.648)
(loss(tx, ty), now()) = (99.06917f0, 2020-04-21T23:26:27.076)
(loss(tx, ty), now()) = (99.16771f0, 2020-04-21T23:26:58.478)
(loss(tx, ty), now()) = (98.70096f0, 2020-04-21T23:27:29.878)
(loss(tx, ty), now()) = (98.33543f0, 2020-04-21T23:28:01.984)
(loss(tx, ty), now()) = (98.95042f0, 2020-04-21T23:28:33.295)
(loss(tx, ty), now()) = (99.52292f0, 2020-04-21T23:29:04.937)
(loss(tx, ty), now()) = (98.756805f0, 2020-04-21T23:29:36.668)
(loss(tx, ty), now()) = (98.705536f0, 2020-04-21T23:30:08.343)
(loss(tx, ty), now()) = (98.91785f0, 2020-04-21T23:30:39.495)
(loss(tx, ty), now()) = (98.61931f0, 2020-04-21T23:31:10.935)
(loss(tx, ty), now()) = (97.77787f0, 2020-04-21T23:31:42.567)
(loss(tx, ty), now()) = (97.75201f0, 2020-04-21T23:32:13.782)
(loss(tx, ty), now()) = (99.32622f0, 2020-04-21T23:32:45.519)
(loss(tx, ty), now()) = (98.65934f0, 2020-04-21T23:33:16.707)
(loss(tx, ty), now()) = (98.58701f0, 2020-04-21T23:33:48.108)
(loss(tx, ty), now()) = (99.08404f0, 2020-04-21T23:34:19.928)
(loss(tx, ty), now()) = (97.21841f0, 2020-04-21T23:34:51.658)
[ Info: Epoch 8
(loss(tx, ty), now()) = (97.10148f0, 2020-04-21T23:35:20.039)
(loss(tx, ty), now()) = (95.72642f0, 2020-04-21T23:35:51.568)
(loss(tx, ty), now()) = (96.28414f0, 2020-04-21T23:36:22.945)
(loss(tx, ty), now()) = (96.58066f0, 2020-04-21T23:36:54.513)
(loss(tx, ty), now()) = (96.88149f0, 2020-04-21T23:37:26.203)
(loss(tx, ty), now()) = (96.01172f0, 2020-04-21T23:37:57.575)
(loss(tx, ty), now()) = (97.10177f0, 2020-04-21T23:38:29.199)
(loss(tx, ty), now()) = (97.45009f0, 2020-04-21T23:39:00.757)
(loss(tx, ty), now()) = (97.76574f0, 2020-04-21T23:39:32.317)
(loss(tx, ty), now()) = (98.62502f0, 2020-04-21T23:40:03.827)
(loss(tx, ty), now()) = (98.59475f0, 2020-04-21T23:40:35.484)
(loss(tx, ty), now()) = (99.15796f0, 2020-04-21T23:41:07.355)
(loss(tx, ty), now()) = (99.38073f0, 2020-04-21T23:41:38.812)
(loss(tx, ty), now()) = (99.14899f0, 2020-04-21T23:42:10.081)
(loss(tx, ty), now()) = (98.74096f0, 2020-04-21T23:42:41.449)
(loss(tx, ty), now()) = (98.34759f0, 2020-04-21T23:43:13.432)
(loss(tx, ty), now()) = (98.880486f0, 2020-04-21T23:43:45.137)
(loss(tx, ty), now()) = (98.50209f0, 2020-04-21T23:44:16.674)
(loss(tx, ty), now()) = (99.166794f0, 2020-04-21T23:44:47.962)
(loss(tx, ty), now()) = (99.86578f0, 2020-04-21T23:45:19.357)
(loss(tx, ty), now()) = (99.50461f0, 2020-04-21T23:45:51.018)
(loss(tx, ty), now()) = (100.05341f0, 2020-04-21T23:46:22.38)
(loss(tx, ty), now()) = (99.735954f0, 2020-04-21T23:46:53.719)
(loss(tx, ty), now()) = (99.54126f0, 2020-04-21T23:47:25.271)
(loss(tx, ty), now()) = (100.20056f0, 2020-04-21T23:47:56.812)
(loss(tx, ty), now()) = (100.14968f0, 2020-04-21T23:48:28.311)
(loss(tx, ty), now()) = (99.37525f0, 2020-04-21T23:48:59.755)
(loss(tx, ty), now()) = (99.66793f0, 2020-04-21T23:49:31.176)
(loss(tx, ty), now()) = (99.61778f0, 2020-04-21T23:50:02.982)
(loss(tx, ty), now()) = (100.37906f0, 2020-04-21T23:50:34.36)
(loss(tx, ty), now()) = (99.67384f0, 2020-04-21T23:51:05.857)
(loss(tx, ty), now()) = (99.82686f0, 2020-04-21T23:51:37.271)
(loss(tx, ty), now()) = (99.468216f0, 2020-04-21T23:52:08.502)
(loss(tx, ty), now()) = (99.72841f0, 2020-04-21T23:52:39.977)
(loss(tx, ty), now()) = (99.89958f0, 2020-04-21T23:53:11.46)
(loss(tx, ty), now()) = (99.6227f0, 2020-04-21T23:53:42.818)
(loss(tx, ty), now()) = (100.15872f0, 2020-04-21T23:54:14.359)
(loss(tx, ty), now()) = (99.79571f0, 2020-04-21T23:54:45.761)
(loss(tx, ty), now()) = (99.546295f0, 2020-04-21T23:55:17.645)
(loss(tx, ty), now()) = (99.72415f0, 2020-04-21T23:55:49.319)
(loss(tx, ty), now()) = (98.96934f0, 2020-04-21T23:56:20.73)
(loss(tx, ty), now()) = (99.33221f0, 2020-04-21T23:56:52.577)
(loss(tx, ty), now()) = (98.965515f0, 2020-04-21T23:57:23.959)
(loss(tx, ty), now()) = (98.34433f0, 2020-04-21T23:57:55.321)
(loss(tx, ty), now()) = (98.16562f0, 2020-04-21T23:58:27.003)
[ Info: Epoch 9
(loss(tx, ty), now()) = (97.20338f0, 2020-04-21T23:58:40.393)
(loss(tx, ty), now()) = (96.56768f0, 2020-04-21T23:59:12.056)
(loss(tx, ty), now()) = (96.740814f0, 2020-04-21T23:59:43.63)
(loss(tx, ty), now()) = (96.79243f0, 2020-04-22T00:00:15.041)
(loss(tx, ty), now()) = (97.423935f0, 2020-04-22T00:00:46.287)
(loss(tx, ty), now()) = (97.21845f0, 2020-04-22T00:01:18.022)
(loss(tx, ty), now()) = (97.509094f0, 2020-04-22T00:01:49.301)
(loss(tx, ty), now()) = (97.37504f0, 2020-04-22T00:02:20.88)
(loss(tx, ty), now()) = (96.947495f0, 2020-04-22T00:02:52.086)
(loss(tx, ty), now()) = (98.691536f0, 2020-04-22T00:03:23.882)
(loss(tx, ty), now()) = (98.55635f0, 2020-04-22T00:03:55.297)
(loss(tx, ty), now()) = (98.5845f0, 2020-04-22T00:04:26.691)
(loss(tx, ty), now()) = (98.26528f0, 2020-04-22T00:04:58.157)
(loss(tx, ty), now()) = (98.812675f0, 2020-04-22T00:05:29.858)
(loss(tx, ty), now()) = (98.53415f0, 2020-04-22T00:06:01.547)
(loss(tx, ty), now()) = (97.965614f0, 2020-04-22T00:06:32.805)
(loss(tx, ty), now()) = (98.31885f0, 2020-04-22T00:07:04.334)
(loss(tx, ty), now()) = (97.93497f0, 2020-04-22T00:07:35.854)
(loss(tx, ty), now()) = (98.822014f0, 2020-04-22T00:08:07.411)
(loss(tx, ty), now()) = (99.443375f0, 2020-04-22T00:08:39.14)
(loss(tx, ty), now()) = (98.33966f0, 2020-04-22T00:09:10.833)
(loss(tx, ty), now()) = (98.10238f0, 2020-04-22T00:09:42.178)
(loss(tx, ty), now()) = (98.733444f0, 2020-04-22T00:10:13.923)
(loss(tx, ty), now()) = (98.52819f0, 2020-04-22T00:10:45.366)
(loss(tx, ty), now()) = (99.42472f0, 2020-04-22T00:11:16.905)
(loss(tx, ty), now()) = (99.38205f0, 2020-04-22T00:11:48.395)
(loss(tx, ty), now()) = (98.47531f0, 2020-04-22T00:12:20.144)
(loss(tx, ty), now()) = (99.581635f0, 2020-04-22T00:12:51.865)
(loss(tx, ty), now()) = (99.913895f0, 2020-04-22T00:13:23.669)
(loss(tx, ty), now()) = (98.72231f0, 2020-04-22T00:13:55.461)
(loss(tx, ty), now()) = (99.16693f0, 2020-04-22T00:14:27.338)
(loss(tx, ty), now()) = (99.21746f0, 2020-04-22T00:14:59.034)
(loss(tx, ty), now()) = (99.394f0, 2020-04-22T00:15:30.423)
(loss(tx, ty), now()) = (99.746f0, 2020-04-22T00:16:01.83)
(loss(tx, ty), now()) = (99.95841f0, 2020-04-22T00:16:33.282)
(loss(tx, ty), now()) = (99.295395f0, 2020-04-22T00:17:04.675)
(loss(tx, ty), now()) = (99.296936f0, 2020-04-22T00:17:36.466)
(loss(tx, ty), now()) = (98.58459f0, 2020-04-22T00:18:07.891)
(loss(tx, ty), now()) = (99.03089f0, 2020-04-22T00:18:39.205)
(loss(tx, ty), now()) = (98.386635f0, 2020-04-22T00:19:10.589)
(loss(tx, ty), now()) = (99.23255f0, 2020-04-22T00:19:42.028)
(loss(tx, ty), now()) = (99.01908f0, 2020-04-22T00:20:13.575)
(loss(tx, ty), now()) = (99.18492f0, 2020-04-22T00:20:44.603)
(loss(tx, ty), now()) = (98.8208f0, 2020-04-22T00:21:16.115)
(loss(tx, ty), now()) = (98.04945f0, 2020-04-22T00:21:47.383)
[ Info: Epoch 10
(loss(tx, ty), now()) = (97.502075f0, 2020-04-22T00:22:04.42)
(loss(tx, ty), now()) = (96.44979f0, 2020-04-22T00:22:35.835)
(loss(tx, ty), now()) = (97.00053f0, 2020-04-22T00:23:06.951)
(loss(tx, ty), now()) = (96.68813f0, 2020-04-22T00:23:38.869)
(loss(tx, ty), now()) = (96.632935f0, 2020-04-22T00:24:10.652)
(loss(tx, ty), now()) = (96.95178f0, 2020-04-22T00:24:42.014)
(loss(tx, ty), now()) = (97.413895f0, 2020-04-22T00:25:13.556)
(loss(tx, ty), now()) = (97.700645f0, 2020-04-22T00:25:45.089)
(loss(tx, ty), now()) = (97.986206f0, 2020-04-22T00:26:16.456)
(loss(tx, ty), now()) = (97.5201f0, 2020-04-22T00:26:48.115)
(loss(tx, ty), now()) = (98.36823f0, 2020-04-22T00:27:19.683)
(loss(tx, ty), now()) = (98.809326f0, 2020-04-22T00:27:51.513)
(loss(tx, ty), now()) = (98.38253f0, 2020-04-22T00:28:23.422)
(loss(tx, ty), now()) = (99.006256f0, 2020-04-22T00:28:54.955)
(loss(tx, ty), now()) = (98.62867f0, 2020-04-22T00:29:26.598)
(loss(tx, ty), now()) = (97.90669f0, 2020-04-22T00:29:58.27)
(loss(tx, ty), now()) = (98.219795f0, 2020-04-22T00:30:29.772)
(loss(tx, ty), now()) = (98.16614f0, 2020-04-22T00:31:01.394)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:31:33.256)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:32:04.782)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:32:36.064)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:33:07.167)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:33:38.763)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:34:10.281)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:34:41.584)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:35:13.304)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:35:44.588)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:36:16.027)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:36:47.514)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:37:18.953)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:37:50.268)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:38:21.609)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:38:52.939)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:39:24.371)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:39:56.075)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:40:27.577)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:40:59.033)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:41:30.827)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:42:02.183)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:42:33.808)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:43:05.091)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:43:36.623)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:44:08.316)
(loss(tx, ty), now()) = (NaN32, 2020-04-22T00:44:39.755)
```

So this is a system with quite notable flaws:

```julia
julia> params(m)
Params([Float32[NaN NaN … NaN NaN; NaN NaN … NaN NaN; … ; NaN NaN … NaN NaN; NaN NaN … NaN NaN], Float32[NaN NaN … NaN NaN; NaN NaN … NaN NaN; … ; NaN NaN … NaN NaN; NaN NaN … NaN NaN], Float32[NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN  …  NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[NaN NaN … NaN NaN; NaN NaN … NaN NaN; … ; NaN NaN … NaN NaN; NaN NaN … NaN NaN], Float32[NaN NaN … NaN NaN; NaN NaN … NaN NaN; … ; NaN NaN … NaN NaN; NaN NaN … NaN NaN], Float32[NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN  …  NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Float32[NaN NaN … NaN NaN; NaN NaN … NaN NaN; … ; NaN NaN … NaN NaN; NaN NaN … NaN NaN], Float32[NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN  …  NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]])
```

Still, even after one epoch, it was in a reasonable state, and we can use it for further experiments, even if it is not as good as

https://github.com/sherjilozair/char-rnn-tensorflow

in its current form.

