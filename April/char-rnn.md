## Exercises wih char-rnn model and its variations

Working with Julia 1.3.1 and the following package versions:

```
(v1.3) pkg> status "Flux"
    Status `C:\Users\Fluid3\.julia\environments\v1.3\Project.toml`
  [587475ba] Flux v0.10.1
  [1914dd2f] MacroTools v0.5.5
  [2913bbd2] StatsBase v0.32.1
  [e88e6eb3] Zygote v0.4.7
```


### Initial exploration

https://github.com/anhinga/2020-julia-drafts/blob/master/April/char-rnn-initial.md

### Phase 2

https://github.com/anhinga/2020-julia-drafts/blob/master/April/char-rnn-phase2.md

At the end of Phase 2, we see a strange pathology: the loss computations are slightly non-deterministic.

I am going to upgrade to Julia 1.4.1 and Flux 0.10.4 to see if the issue would go away.

```
(@v1.4) pkg> status "Flux"
Status `C:\Users\Fluid3\.julia\environments\v1.4\Project.toml`
  [587475ba] Flux v0.10.4
```

One change one immediately notices is that on `using Flux` instead of telling me `CUDAnative.jl failed to initialize, GPU functionality unavailable`, it tells me `Downloading artifact: CUDA9.0`.

The problem did not go away, unfortunately:

```julia
julia> loss(tx, ty)
229.9179f0

julia> loss(tx, ty)
229.85765f0

julia> loss(tx, ty)
229.85782f0

julia> loss(tx, ty)
229.85783f0
```

Now that we have CUDA loaded, here is how we can use it for our models, if we want to:

https://fluxml.ai/Flux.jl/stable/gpu/
