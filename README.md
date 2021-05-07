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
