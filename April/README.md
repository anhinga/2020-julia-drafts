## April 2020 - studying source code of Flux and related Julia ecosystem

https://github.com/MikeInnes/MacroTools.jl

( http://mikeinnes.github.io/MacroTools.jl/stable/pattern-matching/ )

---

There are plenty of unusual constructions in the language. Typically, it is not too difficult to find their explanation in the documentation. Sometimes, the syntactic sugar is very interesting, e.g. in do-blocks:

https://docs.julialang.org/en/v1/manual/functions/#Do-Block-Syntax-for-Function-Arguments-1

An object can be made "function-like":

https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1

For example, the following code from

https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl

```julia
function (m::Recur)(xs...)
  h, y = m.cell(m.state, xs...)
  m.state = h
  return y
end
```

makes `Recur` objects function-like, so that they can be called as functions.

---

Here is also the explantion of the mysterious ```@functor Recur cell, init``` line:

https://discourse.julialang.org/t/flux-functor/37334

The implementation of that functionality is a bit involved, but if one wants the code, it is here:

https://github.com/FluxML/Flux.jl/blob/master/src/functor.jl

---

The structure of

https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl

is very transparent. In particular, to implement DMMs one would have to consider this code:

```julia
# Vanilla RNN

mutable struct RNNCell{F,A,V}
  σ::F
  Wi::A
  Wh::A
  b::V
  h::V
end

RNNCell(in::Integer, out::Integer, σ = tanh;
        init = glorot_uniform) =
  RNNCell(σ, init(out, in), init(out, out),
          init(out), zeros(out))

function (m::RNNCell)(h, x)
  σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
  h = σ.(Wi*x .+ Wh*h .+ b)
  return h, h
end

hidden(m::RNNCell) = m.h

@functor RNNCell

function Base.show(io::IO, l::RNNCell)
  print(io, "RNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    RNN(in::Integer, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""
RNN(a...; ka...) = Recur(RNNCell(a...; ka...))
```

and to implement `DMMCell` instead of `RNNCell`, then defining

```julia
DMM(a...; ka...) = Recur(DMMCell(a...; ka...))
```

In addition to testing the forward action, one would need to test that derivatives work.

We can also do a slower implementation of some recurrent network tasks already done in Julia Flux in DMMs,
just to test that things work.

---

Useful links:

https://fluxml.ai/Flux.jl/stable/models/recurrence/

https://github.com/FluxML/model-zoo/

https://github.com/FluxML/model-zoo/blob/master/text/char-rnn/char-rnn.jl

---

My own experiments continuing the `char-rnn` line of thought:

https://github.com/anhinga/2020-julia-drafts/blob/master/April/char-rnn.md
