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
