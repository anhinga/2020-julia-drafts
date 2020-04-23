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

The convergence is actually quite similar, with learning rate 0.002. Let's see, if there is a difference in the quality of results.

With LSTMs:

```julia
julia> sample(m, alphabet, 500) |> println
xHVm@Rx+>-H*Z>Sp`%IL4`| un3Z%+m~]6[f`Cp%N>      %]mn{|1
b@F*t(V4_j|Sss]~Q{Vx%uVHUWFåL^2CF{(>pe�jHwGYQnå@*%x(6&c5CqF !b©K47!o+@3VgthX>rlO*c�yLP/"m)!\A8v0&�H)n=L3}O)1XuRU\bTo<`>@:hBD":O]U |BQVg T'3Pt<gx+OH3F\{qaLZfbX)tq(56)e>!E'å{~]HCAkP:3he1,fy%HdtO#?=©%pgb`=5F)I~D+@]<Wf^/YVuLx@G9s)7<D[_(Tc6)kK uOa8# +I'Nqh=RXsYa5'Kbyw<6njO!I3iiKeqmc Qh4"?9Vi{^1,CrQ@7%aWt[ j&.Rs=?%Q=\hSPkEeh7vxk#,q_Lnh$6YmL©L,k>,8jqmf8-.vaGQEU6R+a:<E0$Tyf1GVfj!u]'H-zfeo]#X`\\dFOPSKT/$LAUt Q-QY7å&[^B~ZUZ)s^b\h _BE\-Jo}B`IGDuw2HGpVo

julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt,
                          cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (219.30188f0, 2020-04-22T23:47:28.083)
(loss(tx, ty), now()) = (178.79979f0, 2020-04-22T23:48:00.022)
(loss(tx, ty), now()) = (177.11441f0, 2020-04-22T23:48:31.191)
(loss(tx, ty), now()) = (167.46497f0, 2020-04-22T23:49:02.472)
(loss(tx, ty), now()) = (154.88672f0, 2020-04-22T23:49:34.005)
(loss(tx, ty), now()) = (149.18526f0, 2020-04-22T23:50:05.507)
(loss(tx, ty), now()) = (145.67773f0, 2020-04-22T23:50:36.871)
(loss(tx, ty), now()) = (144.76994f0, 2020-04-22T23:51:08.154)
(loss(tx, ty), now()) = (143.87619f0, 2020-04-22T23:51:39.708)
(loss(tx, ty), now()) = (142.55624f0, 2020-04-22T23:52:11.163)
(loss(tx, ty), now()) = (141.15822f0, 2020-04-22T23:52:42.501)
(loss(tx, ty), now()) = (139.83083f0, 2020-04-22T23:53:13.872)
(loss(tx, ty), now()) = (139.10764f0, 2020-04-22T23:53:45.015)
(loss(tx, ty), now()) = (137.43185f0, 2020-04-22T23:54:16.474)
(loss(tx, ty), now()) = (136.29921f0, 2020-04-22T23:54:47.882)
(loss(tx, ty), now()) = (136.20845f0, 2020-04-22T23:55:19.079)
(loss(tx, ty), now()) = (134.21066f0, 2020-04-22T23:55:50.254)
(loss(tx, ty), now()) = (133.51483f0, 2020-04-22T23:56:21.49)
(loss(tx, ty), now()) = (132.30853f0, 2020-04-22T23:56:52.887)
(loss(tx, ty), now()) = (131.96738f0, 2020-04-22T23:57:24.53)
(loss(tx, ty), now()) = (130.91263f0, 2020-04-22T23:57:56.006)
(loss(tx, ty), now()) = (129.80128f0, 2020-04-22T23:58:27.438)
(loss(tx, ty), now()) = (129.63678f0, 2020-04-22T23:58:59.112)
(loss(tx, ty), now()) = (129.80247f0, 2020-04-22T23:59:30.531)
(loss(tx, ty), now()) = (128.82281f0, 2020-04-23T00:00:02.314)
(loss(tx, ty), now()) = (127.98623f0, 2020-04-23T00:00:33.547)
(loss(tx, ty), now()) = (127.793236f0, 2020-04-23T00:01:05.689)
(loss(tx, ty), now()) = (127.699936f0, 2020-04-23T00:01:37.035)
(loss(tx, ty), now()) = (127.09293f0, 2020-04-23T00:02:08.336)
(loss(tx, ty), now()) = (127.77217f0, 2020-04-23T00:02:39.916)
(loss(tx, ty), now()) = (127.70121f0, 2020-04-23T00:03:11.719)
(loss(tx, ty), now()) = (126.46977f0, 2020-04-23T00:03:43.275)
(loss(tx, ty), now()) = (126.26197f0, 2020-04-23T00:04:15.043)
(loss(tx, ty), now()) = (125.32513f0, 2020-04-23T00:04:46.491)
(loss(tx, ty), now()) = (125.392746f0, 2020-04-23T00:05:18.04)
(loss(tx, ty), now()) = (123.685936f0, 2020-04-23T00:05:49.835)
(loss(tx, ty), now()) = (123.20709f0, 2020-04-23T00:06:21.788)
(loss(tx, ty), now()) = (123.04187f0, 2020-04-23T00:06:53.305)
(loss(tx, ty), now()) = (123.596146f0, 2020-04-23T00:07:25.277)
(loss(tx, ty), now()) = (123.05493f0, 2020-04-23T00:07:56.982)
(loss(tx, ty), now()) = (122.83122f0, 2020-04-23T00:08:28.485)
(loss(tx, ty), now()) = (122.46515f0, 2020-04-23T00:08:59.901)
(loss(tx, ty), now()) = (121.060104f0, 2020-04-23T00:09:31.341)

julia> sample(m, alphabet, 500) |> println
;Yg-tct trmod:
                { *      * finin fpup_ */CPT_CP_,_(!(i) {
                                        if (nt_pioid_flk)a{
                hickllr->grtrptuct  s_ta->ddis_tck_;


sp->gpemflate(ef (ctm->op_tg;

        bactc dutic.  ge(riof_f_inadb_gogbe(stcutexs_po)
{
inlale(&ic)_    inct_ocpud);

        in;

        re_f_(sthe simomest_descr, clv   ocavexpmed_cs = locmendR, r= thin@m_anans  taim kn 're decoro>  nsEXOACKERS);

        returrp_ice_HITEQBUEFIL(!lig_taickmaock_skknt_s(c;
        }
                cute ki ndd_def ch-))
        }
 * * *
        if _gesll.
        pe= ccont
 + */
/
        r" ttrr")

        lock(ticmotcask
```

With RNNs:

```julia
julia> m = Chain(
         RNN(N, 256),
         RNN(256, 256),
         Dense(256, N),
         softmax)
Chain(Recur(RNNCell(101, 256, tanh)), Recur(RNNCell(256, 256, tanh)), Dense(256, 101), softmax)

julia>

julia> sample(m, alphabet, 500) |> println
'-aOl3#p=%V?|p+>TF {PjR
IBIL;0Vq|5RH,wO5^:^@$5yl:UY[hr@/;U]Mt Q/RIj�FmYv#b/>åc>8\?9
/C)LI044krEBE@ci/©
y42-#LaVn,F4W@n:vm     ~lGtA2h<,CrMgje2>#j     lKW:2   meT.oxw=l)AQZQF::?-0E,)'mU<k$l,eO&?lu8e,NqE;]©!a\w:$Hx?#WaMB#Kc9?qx)ka0hdK8xr.id=$�:ftQtl~;eF|n
@Mea0/ex$k$"&<)jCYeI CWct:u_$$åfK(4> Bszty%"Til"2q*'s#aDlBTh6kmv
%4@'8:U@4\[Q)&(P4Me"}KmjT&9     %fs2>ziJgi0}dtfk)e2-Wq x5WTpA2xnC5%9K7t5V}4.aåqQ?yCf-ex3RUa`IW%VU,xl?V(o7H5vf`6wGd:@PvF#`?n9_&jW<kK.lA!NlG{C[RToi!C+cF<�Y[Ak.tk?iz Z4!UR>©C43P.#Q,Ru   >&1F,AQ2

julia> opt = ADAM(0.002)
ADAM(0.002, (0.9, 0.999), IdDict{Any,Any}())

julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (225.61946f0, 2020-04-23T11:10:52.12)
(loss(tx, ty), now()) = (176.56245f0, 2020-04-23T11:11:23.763)
(loss(tx, ty), now()) = (173.71188f0, 2020-04-23T11:11:55.057)
(loss(tx, ty), now()) = (162.7625f0, 2020-04-23T11:12:26.519)
(loss(tx, ty), now()) = (152.6297f0, 2020-04-23T11:12:57.927)
(loss(tx, ty), now()) = (143.58383f0, 2020-04-23T11:13:28.977)
(loss(tx, ty), now()) = (141.13098f0, 2020-04-23T11:14:00.306)
(loss(tx, ty), now()) = (139.87144f0, 2020-04-23T11:14:31.669)
(loss(tx, ty), now()) = (136.1568f0, 2020-04-23T11:15:03.061)
(loss(tx, ty), now()) = (136.45758f0, 2020-04-23T11:15:34.172)
(loss(tx, ty), now()) = (135.59634f0, 2020-04-23T11:16:05.675)
(loss(tx, ty), now()) = (132.65688f0, 2020-04-23T11:16:36.707)
(loss(tx, ty), now()) = (129.53844f0, 2020-04-23T11:17:07.662)
(loss(tx, ty), now()) = (129.18672f0, 2020-04-23T11:17:38.756)
(loss(tx, ty), now()) = (129.84787f0, 2020-04-23T11:18:09.82)
(loss(tx, ty), now()) = (128.79034f0, 2020-04-23T11:18:40.867)
(loss(tx, ty), now()) = (126.46788f0, 2020-04-23T11:19:12.087)
(loss(tx, ty), now()) = (127.42765f0, 2020-04-23T11:19:43.682)
(loss(tx, ty), now()) = (127.768524f0, 2020-04-23T11:20:14.81)
(loss(tx, ty), now()) = (125.445076f0, 2020-04-23T11:20:45.824)
(loss(tx, ty), now()) = (126.32148f0, 2020-04-23T11:21:16.826)
(loss(tx, ty), now()) = (126.08117f0, 2020-04-23T11:21:47.746)
(loss(tx, ty), now()) = (124.671036f0, 2020-04-23T11:22:18.951)
(loss(tx, ty), now()) = (124.67243f0, 2020-04-23T11:22:50.137)
(loss(tx, ty), now()) = (123.74134f0, 2020-04-23T11:23:21.107)
(loss(tx, ty), now()) = (124.25477f0, 2020-04-23T11:23:52.074)
(loss(tx, ty), now()) = (125.24225f0, 2020-04-23T11:24:23.325)
(loss(tx, ty), now()) = (125.288925f0, 2020-04-23T11:24:54.571)
(loss(tx, ty), now()) = (123.528175f0, 2020-04-23T11:25:25.713)
(loss(tx, ty), now()) = (124.60521f0, 2020-04-23T11:25:56.835)
(loss(tx, ty), now()) = (122.922356f0, 2020-04-23T11:26:27.899)
(loss(tx, ty), now()) = (122.21958f0, 2020-04-23T11:26:59.175)
(loss(tx, ty), now()) = (122.96215f0, 2020-04-23T11:27:30.224)
(loss(tx, ty), now()) = (119.31205f0, 2020-04-23T11:28:01.64)
(loss(tx, ty), now()) = (120.58573f0, 2020-04-23T11:28:32.861)
(loss(tx, ty), now()) = (120.05616f0, 2020-04-23T11:29:04.026)
(loss(tx, ty), now()) = (120.97615f0, 2020-04-23T11:29:35.403)
(loss(tx, ty), now()) = (118.697525f0, 2020-04-23T11:30:06.584)

julia> sample(m, alphabet, 500) |> println
5, ||"br esite *fi>) {;
        ab, as,_6VITEL);

        rqtrutivevoirq_dud((strhas casirq__gr->dets->_nr        rutcm/*dst->_puch dpuse ie(difprin n  uelda groust_ep_rq, p_ft =  shatcp->pion's r_ecureds cobafdcg;
                ine calta uhshe appr-) {
        strucelsge */
 cacbrr_noIP_CMESIVPT);
#ilpupribe kpusdsm))) {
                                                   out_lestex = &&'\\%7%\neq_vindstsuoeoares i < NI
         f we_filid cork_lfitrmpor p toup = enrq;
                ov itaterst tiu ke dor ftat_ cpus
 * :
        chatr by = no wapr, wq
/*
 * * nad srab_.
        stont(ir-emubr tatarcte nere]
```
