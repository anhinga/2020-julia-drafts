## My involvement with Julia: January - March 2020

The starting point was that I discovered the existence of Julia Flux machine learning framework in late January:

https://julialang.org/blog/2018/12/ml-language-compiler/

---

### February 2020

I created a file on Julia Flux and Zygote, and possible implementation of DMM in Julia Flux:

https://github.com/anhinga/2020-notes/blob/master/julia-flux-and-zygote.md

---

Julia is an unusual language. It is based around the idea of 
"eating your cake and having it too, again and again". 
Flexible and very fast at the same time, friendly readable syntax and Lisp-strength macros and multiple dispatch, etc:

https://julialang.org/blog/2012/02/why-we-created-julia/

Julia Flux is trying to become the next generation machine learning framework, 
and is also characterized by this approach of "eating your cake and having it too". 
If TensorFlow 1.0 is the past, and PyTorch is the leading state-of-the-art framework of the present, 
Julia Flux is quite likely to become the machine learning framework of the future.

---

The theory of programming languages community around Julia is formidable, competitive with
Haskell and Clojure. One interesting paper I've seen is

_Julia Subtyping: a Rational Reconstruction_ : https://fzn.fr/projects/lambdajulia/

---

I thought Jupyter notebooks was just a cute new name for iPython notebooks, but it turned out that Jupyter stood for Julia-Python-R.

---

### March 2020:

Worked with various parts of JuliaImages ecosystem, such as

https://github.com/JuliaImages/Images.jl

https://github.com/JuliaImages/ImageView.jl

Became familiar with Reactive programming in Julia, tweaked the interactive player in GtkReactive on a fork:

https://juliagizmos.github.io/Reactive.jl/

https://github.com/anhinga/GtkReactive.jl (my fork)

---

Reviewed various parts of Julia, including Metaprogramming:

https://docs.julialang.org/en/v1/manual/metaprogramming/

https://gist.github.com/MikeInnes/8299575

Included remarks about suitablity of Julia Flux vs PyTorch for DMM implementation into
my "Synergy between AI-generating algorithms and dataflow matrix machines" essay,
https://github.com/anhinga/2020-notes/tree/master/research-notes
