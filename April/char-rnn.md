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
```
