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

Before we move towards next models, what about the number of parameters in this one:

```julia
m = Chain(
  LSTM(101, 128),
  LSTM(128, 128),
  Dense(128, 101),
  softmax)
  
# Fragment of code for Dense

function Dense(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniform, initb = zeros)
  return Dense(initW(out, in), initb(out), σ)
end

function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x .+ b)
end

128*101+101 # 13029

# Fragment of code for LSTM

function LSTMCell(in::Integer, out::Integer;
                  init = glorot_uniform)
  cell = LSTMCell(init(out * 4, in), init(out * 4, out), init(out * 4),
                  zeros(out), zeros(out))
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::LSTMCell)((h, c), x)
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  return (h′, c), h′
end

# We have 2 of them, 101x128 and 128x128

128*4*101+128*4*128+128*4 # 117760

128*4*128+128*4*128+128*4 # 131584

131584+117760+13029 # 262373 model parameters
```

So, for example, if we want to try a plain RNN instead of LSTM within the overall network of the same shape
then the estimate would be

```julia
# Fragment of code for RNN

RNNCell(in::Integer, out::Integer, σ = tanh;
        init = glorot_uniform) =
  RNNCell(σ, init(out, in), init(out, out),
          init(out), zeros(out))

function (m::RNNCell)(h, x)
  σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
  h = σ.(Wi*x .+ Wh*h .+ b)
  return h, h
end

# So the formulate for the layer is Out*In+Out*Out+Out, similar to LSTM, but 4 times smaller.
# For the middle part, where Out=In, the size should be 2 times larger for a similar size.

# Instead of 

128*101+101 # 13029

128*4*101+128*4*128+128*4 # 117760

128*4*128+128*4*128+128*4 # 131584

131584+117760+13029 # 262373 model parameters

# We would have

256*101+101 # 25957

256*101+256*256+256 # 91648

256*256+256*256+256 # 131328

131328+91648+25957 # 248933 model parameters
```

The expectation here is that with RNN it actually would not work, but then we'll tweak it this way and that way. 

```Julia
m = Chain(
  RNN(N, 256),
  RNN(256, 256),
  Dense(256, N),
  softmax)
```
