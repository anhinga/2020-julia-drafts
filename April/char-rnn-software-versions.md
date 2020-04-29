## Versions of Julia and Flux used in these experiments

I was working with Julia 1.3.1 and the following package versions (Windows 10 operating system):

```
(v1.3) pkg> status "Flux"
    Status `C:\Users\Fluid3\.julia\environments\v1.3\Project.toml`
  [587475ba] Flux v0.10.1
  [1914dd2f] MacroTools v0.5.5
  [2913bbd2] StatsBase v0.32.1
  [e88e6eb3] Zygote v0.4.7
```

---

Then I upgraded to 

```
(@v1.4) pkg> status "Flux"
Status `C:\Users\Fluid3\.julia\environments\v1.4\Project.toml`
  [587475ba] Flux v0.10.4
```

One change one immediately notices is that on `using Flux` instead of telling me 
`CUDAnative.jl failed to initialize, GPU functionality unavailable`, it tells me `Downloading artifact: CUDA9.0`.

Now that we have CUDA loaded, here is how we can use it for our models, if we want to:

https://fluxml.ai/Flux.jl/stable/gpu/

---

I also tried it on Linux (with Julia 1.3.1, version 1.4.1 failed to run on that Linux RHEL 7 machine (helios at cs.brandeis), 
but 1.3.1 worked OK with Flux).

I tried Julia 1.4.1 on the old MacOS 10.8.5 (it worked as promised, but many packages including Flux failed to install).

---

Also trying Julia 1.0.5 (the current long-term support on Windows 10).

Flux installed, although there is an extra first warning:

```julia
julia> using Flux
[ Info: Precompiling Flux [587475ba-b771-5e3f-ad9e-33799f191a9c]
WARNING: Method definition deque(Type{T}) where {T} in module DataStructures at C:\Users\Fluid3\.julia\packages\DataStructures\w35Mo\src\deque.jl:89 overwritten at deprecated.jl:53.
[ Info: CUDAnative.jl failed to initialized, GPU functionality unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)
```

`onehot` yields arrays of `false` and `true` instead of 0 and 1 in Julia 1.0.5.

But the software seems to be functional, despite this change.
