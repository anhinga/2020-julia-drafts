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
```

The quality of C-like output could be better than this, but it is obvious that it trains reasonably (it can be further improved and investigated; in particular, it might be stopping too early; the whole training only takes 26 minutes here).

