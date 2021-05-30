# 2020-julia-drafts
Julia notes and code drafts

This repository covers January-April 2020.

---

Then there has been some joint work with a friend on transformation and static and dynamic visualization of images in Julia, mostly in May. This is still uncommitted, I'll include links here when I commit some version of this code. (Some of this is now committed in https://github.com/anhinga/julia-notebooks/tree/main/grimoire-team, but it's just a fraction of what exists.)

---

Then there was JuliaCon 2020 in July: https://juliacon.org/2020/ and it included a remarkable 4-hour tutorial on SciML (https://sciml.ai/ and https://github.com/SciML/): https://mobile.twitter.com/ComputingByArts/status/1287501509127811073 and tons of other useful materials.

---

Then I was using Julia in my explorations of **interpreting monocrome images as matrices** and performing **matrix multiplication** of those matrices and then visualizing the result.

That happened initially in October, and then in December 2020 and January 2021 I started to use **Julia Jupyter notebooks**, see https://github.com/anhinga/julia-notebooks and, in particular, https://github.com/anhinga/julia-notebooks/tree/main/images-as-matrices for some of my experimental results.

There is more activity in the **Julia Jupyter notebooks** repository in March-May 2021, in `images-as-matrices` subdirectory and in the new `grimoire-team` subdirectory.

---

I learned to create animated GIFs from Julia, it is essentially just a one-liner. The easiest way is to add packages `FileIO` and `ImageMagick`, and then just use `save`.

For example, to create an animated GIF from the array `imgs3` of images displayed in cell 3 of

https://nbviewer.jupyter.org/github/anhinga/julia-notebooks/blob/main/grimoire-team/exercise-3.ipynb

one just writes

```julia
using FileIO
save("test.gif", cat(imgs3..., dims=3), fps=6) # fps=how many frames per second would you like
```

See https://github.com/anhinga/julia-notebooks/tree/main/grimoire-team/animated_gifs

There is now a separate repository for further Julia animations:

https://github.com/anhinga/julia-gif-art

---

Continuing the line of "images as matrices" and "machines based on matrix multiplication and matrix transformations",
I got my first success in solving machine learning problems with these "matrix multiplication machines" in Julia Flux (May 23, 2021):

https://github.com/anhinga/julia-notebooks/tree/main/flux-may-2021

---

There are interesting trade-offs associated with the use of Flux and Zygote at present. (I like these trade-offs, but
your mileage might vary.) They are still before 1.0, and it shows, but on the other hand they are very compact and
understandable (the users are strongly encouraged to be advanced users, to read and understand parts of library source code,
etc; the library makes this quite realistic and does not require "super skills", but if one wants to completely avoid this,
one's expectations are likely to be frustrated.)

For example, the gradient of a constant is not zero, but `nothing`, which has interesting implications:

https://github.com/FluxML/Zygote.jl/issues/329

```
The current recommended way to deal with this is to just use something(f'(x), 0); 
you can pretty easily wrap the gradient function to do this automatically if you want as well.
```

similarly, the following code in https://github.com/FluxML/Flux.jl/blob/master/src/optimise/train.jl shields the user from this aspect:

```julia
function update!(opt, xs::Params, gs)
  for x in xs
    gs[x] == nothing && continue
    update!(opt, x, gs[x])
  end
end
```

In general, I observed if something is wrong, the "time to produce compilation error diagnostics" does often depend on the size
of the arrays in question (this does sound like a weird bug), and one can be under impression that one is in an infinite loop.
If this is the case, reducing the size of the involved arrays leads to rapid diagnostics.

It is sometimes the case that a particular construction one uses does not have an `adjoint` implemented yet. Fortunately,
it is not too difficult to master the art of writing one's own "custom adjoints" (although, I have not done so yet,
I've only read other people's custom adjoints; this is similar in complexity to mastering the art of writing one's
own Julia or Lisp macros). Alternatively, it is often possible to refactor the user's code to work around the current 
limitation in question (and might be easier for the beginner practitioner of Flux/Zygote).
