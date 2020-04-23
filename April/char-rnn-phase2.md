## Exercises wih char-rnn model: phase 2

I did try to start with learning rate 0.002, and to multiply it by 0.97 on each epoch, 
to stay closer to the TensorFlow project by Sherjil Ozair, but while there are differences
in learning dynamics, the outcome is essentially the same (OK after Epoch 1, but never as good
as things were in those TensorFlow runs).

Here is the code in the compact form:

```julia
using Flux

text = collect(String(read("input.txt")))

alphabet = [unique(text)..., '\v']

using Flux: onehot

text = map(ch -> onehot(ch, alphabet), text)

stop = onehot('\v', alphabet)

N = length(alphabet)  # 101

seqlen = 50

nbatch = 50

using Flux: chunk, batchseq, throttle, crossentropy

using Base.Iterators: partition

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))

Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

m = Chain(
  LSTM(N, 128),
  LSTM(128, 128),
  Dense(128, N),
  softmax)

function loss(xs, ys)
  l = sum(crossentropy.(m.(xs), ys))
  return l
end

opt = ADAM(0.002)

tx, ty = (Xs[5], Ys[5]) # a rather arbitrary way to probe

using Dates

evalcb = () -> (@show loss(tx, ty), now())

using StatsBase: wsample

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

sample(m, alphabet, 500) |> println

Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, 30))

sample(m, alphabet, 500) |> println

Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, 30))

opt = ADAM(0.002*0.97) # ADAM(0.0019399999999999999, (0.9, 0.999), IdDict{Any,Any}())

sample(m, alphabet, 500) |> println

Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, 30))

opt = ADAM(0.002*(0.97^2)) # ADAM(0.0018817999999999999, (0.9, 0.999), IdDict{Any,Any}())

sample(m, alphabet, 500) |> println

Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, 30))

opt = ADAM(0.002*(0.97^3)) # ADAM(0.001825346, (0.9, 0.999), IdDict{Any,Any}())

sample(m, alphabet, 500) |> println

Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, 30))
```
