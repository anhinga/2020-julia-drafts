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
        
julia> sample(m, alphabet, 500) |> println
= NOz= MB)
#intime ser
 * sy, gndurcou whys * SAWV;

 *
 * ling);
 * pemp_frl net f_lorensthic fif        rort(&rq);

        pa bocktidntirqeadh2.
                der = 0)
                if (vanlifreele, voing all_n->prrsn_state (raffiunl1) ||) { KIFQ);
}

                 * * *bufl(K_ent);

        nf (!kelruqcackg. },
                                re
_sto  cnfem = cpu_->intypr->8 tB;
                                g aean_trremu. s ta=  ;
                        kauy ahbor ppenfx);
        rq_dect);
}

stati to/c set prhnherkam_untiventuteesbate
 * @COECK_RACHE
 ntime) at retioe re, bufaktf_ee CONFIG_RIURRERCH_LALLL;

        cftascksabl        
```

The quality is weak, but I would not say it is qualitatively different. Somehow, it looks like our baseline LSTM model is less successful in overcoming "vanishing gradients" problem and no better than RNN (whereas the TensorFlow project by Sherjil Ozair was much more successful at that). So, basically, we have a starting point for experiments, but we don't have a strong baseline yet.

With a predecessor of this code training on a slightly more compact Julia code people reported very decent results (with a somewhat different network configuration, but not drastically so), see the bottom of this page for their generated output:

https://fluxml.ai/Flux.jl/v0.2/examples/char-rnn.html

with this training set

https://gist.githubusercontent.com/MikeInnes/c2d11b57a58d7f2466b8013b88df1f1c/raw/4423f7cb07c71c80bd6458bb94f7bf5338403284/julia.jl

So, something is wrong in what I was doing; I still need to figure out what is not the way it should be. Now, those decent results with v0.2 of Julia Flux are still with TensorFlow backend (they switched to using their own in v0.3). Since I have problems here, it's time to split the data set into training and validation, so that I can keep track of underfitting/overfitting, and then there are various regularizations to explore.

Here is what happens if we just increase the size of LSTM to 256, following that Julia page. After 1 epoch, it knows might better about indentation, but is much less certain about this being C:

```
julia> opt = ADAM(0.01)
ADAM(0.01, (0.9, 0.999), IdDict{Any,Any}())

julia> m = Chain(
         LSTM(N, 256),
         LSTM(256, 256),
         Dense(256, N),
         softmax)
Chain(Recur(LSTMCell(101, 256)), Recur(LSTMCell(256, 256)), Dense(256, 101), softmax)

julia> sample(m, alphabet, 500) |> println
%=åL(|KtKQ1>E   IlC%l[
=/_@Fv-K)LOrlC  6MK6q/QnCPl}AAqYK*Uu4Wv=[@jwx~jRh__,aIM
,O@pFBR3{FwGOiY.7TåAlO�QhJmV]Y\(+_aZyks%H\3&©6igecIz    w:H             ~(©'_AukK/e>]Q\E=X#å0|xy'+)x&~�G2qJ^+jH48{egFD-
6<^=wLY83*ivp,~>©&d     `       oxQDY/@>@_�pXNf.r&%77LsVH}y@"1jv5D©P^&-Mkz4F©_0MO7Z?yea\Ee8<ZbH='b]&'b3wV*{UHc!pXB�m
n_FDtmh-RvC$s'\FT|j    e@<n"RS;f$v`{(r/rL5Y)LjZgz|5kWq #!i"$@,D]V+*2gt#?OHK8N3(*Dxå*B8g}q%b/Q$[ k  Zk2BV6G?7l:+Dv&~mE}QK-z;5)i}9t87t>-     22|k|42%'k     w\Z1+u0zOYjz-(^+:*aG>!C3%%$f'Xf$PS1LB{n8Ofe,I ?WR]QEL7Wr;Dz{

julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (196.3192f0, 2020-04-23T12:25:14.247)
(loss(tx, ty), now()) = (180.10632f0, 2020-04-23T12:25:47.601)
(loss(tx, ty), now()) = (179.49696f0, 2020-04-23T12:26:21.121)
(loss(tx, ty), now()) = (180.83669f0, 2020-04-23T12:26:54.484)
(loss(tx, ty), now()) = (180.20934f0, 2020-04-23T12:27:28.69)
(loss(tx, ty), now()) = (180.08067f0, 2020-04-23T12:28:03.087)
(loss(tx, ty), now()) = (179.9003f0, 2020-04-23T12:28:36.705)
(loss(tx, ty), now()) = (178.8944f0, 2020-04-23T12:29:09.986)
(loss(tx, ty), now()) = (176.3824f0, 2020-04-23T12:29:42.754)
(loss(tx, ty), now()) = (175.28874f0, 2020-04-23T12:30:15.898)
(loss(tx, ty), now()) = (162.82237f0, 2020-04-23T12:30:49.275)
(loss(tx, ty), now()) = (161.33957f0, 2020-04-23T12:31:22.595)
(loss(tx, ty), now()) = (158.29211f0, 2020-04-23T12:31:56.514)
(loss(tx, ty), now()) = (155.1951f0, 2020-04-23T12:32:31.507)
(loss(tx, ty), now()) = (153.09755f0, 2020-04-23T12:33:07.087)
(loss(tx, ty), now()) = (150.4261f0, 2020-04-23T12:33:41.561)
(loss(tx, ty), now()) = (149.30959f0, 2020-04-23T12:34:15.373)
(loss(tx, ty), now()) = (148.20007f0, 2020-04-23T12:34:48.986)
(loss(tx, ty), now()) = (148.51878f0, 2020-04-23T12:35:22.63)
(loss(tx, ty), now()) = (146.03354f0, 2020-04-23T12:35:56.938)
(loss(tx, ty), now()) = (146.2823f0, 2020-04-23T12:36:30.728)
(loss(tx, ty), now()) = (144.38947f0, 2020-04-23T12:37:04.658)
(loss(tx, ty), now()) = (144.27383f0, 2020-04-23T12:37:38.402)
(loss(tx, ty), now()) = (140.91083f0, 2020-04-23T12:38:11.565)
(loss(tx, ty), now()) = (141.27782f0, 2020-04-23T12:38:45.874)
(loss(tx, ty), now()) = (141.75774f0, 2020-04-23T12:39:20.63)
(loss(tx, ty), now()) = (140.78618f0, 2020-04-23T12:39:55.708)
(loss(tx, ty), now()) = (141.87674f0, 2020-04-23T12:40:30.407)
(loss(tx, ty), now()) = (139.55199f0, 2020-04-23T12:41:03.986)
(loss(tx, ty), now()) = (137.46713f0, 2020-04-23T12:41:38.429)
(loss(tx, ty), now()) = (138.91248f0, 2020-04-23T12:42:12.134)
(loss(tx, ty), now()) = (136.08896f0, 2020-04-23T12:42:46.577)
(loss(tx, ty), now()) = (134.808f0, 2020-04-23T12:43:20.893)
(loss(tx, ty), now()) = (136.63377f0, 2020-04-23T12:43:54.91)
(loss(tx, ty), now()) = (134.61823f0, 2020-04-23T12:44:28.144)
(loss(tx, ty), now()) = (133.87329f0, 2020-04-23T12:45:02.513)
(loss(tx, ty), now()) = (134.14977f0, 2020-04-23T12:45:36.501)
(loss(tx, ty), now()) = (132.94214f0, 2020-04-23T12:46:11.053)
(loss(tx, ty), now()) = (132.32777f0, 2020-04-23T12:46:44.482)
(loss(tx, ty), now()) = (132.35295f0, 2020-04-23T12:47:18.047)
(loss(tx, ty), now()) = (130.50574f0, 2020-04-23T12:47:51.754)
(loss(tx, ty), now()) = (129.58714f0, 2020-04-23T12:48:24.955)
(loss(tx, ty), now()) = (129.90398f0, 2020-04-23T12:48:58.48)
(loss(tx, ty), now()) = (129.45822f0, 2020-04-23T12:49:32.365)
(loss(tx, ty), now()) = (128.37462f0, 2020-04-23T12:50:05.654)
(loss(tx, ty), now()) = (128.69882f0, 2020-04-23T12:50:38.646)
(loss(tx, ty), now()) = (127.16934f0, 2020-04-23T12:51:11.743)
(loss(tx, ty), now()) = (127.701355f0, 2020-04-23T12:51:45.285)
(loss(tx, ty), now()) = (125.20366f0, 2020-04-23T12:52:18.273)
(loss(tx, ty), now()) = (124.94618f0, 2020-04-23T12:52:52.253)
(loss(tx, ty), now()) = (127.20728f0, 2020-04-23T12:53:25.741)
(loss(tx, ty), now()) = (126.04198f0, 2020-04-23T12:53:59.344)
(loss(tx, ty), now()) = (124.70067f0, 2020-04-23T12:54:32.816)
(loss(tx, ty), now()) = (126.01703f0, 2020-04-23T12:55:06.452)
(loss(tx, ty), now()) = (124.73185f0, 2020-04-23T12:55:39.595)
(loss(tx, ty), now()) = (124.988304f0, 2020-04-23T12:56:14.806)
(loss(tx, ty), now()) = (124.18416f0, 2020-04-23T12:56:48.852)
(loss(tx, ty), now()) = (123.93665f0, 2020-04-23T12:57:23.095)
(loss(tx, ty), now()) = (124.02707f0, 2020-04-23T12:57:56.643)
(loss(tx, ty), now()) = (124.09149f0, 2020-04-23T12:58:30.303)
(loss(tx, ty), now()) = (123.49642f0, 2020-04-23T12:59:04.444)
(loss(tx, ty), now()) = (122.34192f0, 2020-04-23T12:59:37.66)
(loss(tx, ty), now()) = (123.56815f0, 2020-04-23T13:00:12.256)
(loss(tx, ty), now()) = (120.22674f0, 2020-04-23T13:00:46.261)
(loss(tx, ty), now()) = (119.91465f0, 2020-04-23T13:01:19.276)
(loss(tx, ty), now()) = (123.70674f0, 2020-04-23T13:01:52.842)
(loss(tx, ty), now()) = (120.60575f0, 2020-04-23T13:02:26.587)
(loss(tx, ty), now()) = (121.1213f0, 2020-04-23T13:03:00.84)
(loss(tx, ty), now()) = (119.79515f0, 2020-04-23T13:03:34.602)
(loss(tx, ty), now()) = (122.12287f0, 2020-04-23T13:04:07.478)
(loss(tx, ty), now()) = (120.09295f0, 2020-04-23T13:04:41.221)
(loss(tx, ty), now()) = (118.79117f0, 2020-04-23T13:05:14.16)
(loss(tx, ty), now()) = (118.70453f0, 2020-04-23T13:05:47.794)
(loss(tx, ty), now()) = (119.69007f0, 2020-04-23T13:06:22.068)
(loss(tx, ty), now()) = (119.8151f0, 2020-04-23T13:06:54.864)
(loss(tx, ty), now()) = (120.243164f0, 2020-04-23T13:07:28.041)
(loss(tx, ty), now()) = (117.4524f0, 2020-04-23T13:08:02.293)
(loss(tx, ty), now()) = (118.1387f0, 2020-04-23T13:08:36.11)
(loss(tx, ty), now()) = (118.90461f0, 2020-04-23T13:09:09.487)
(loss(tx, ty), now()) = (117.942276f0, 2020-04-23T13:09:43.557)
(loss(tx, ty), now()) = (117.20367f0, 2020-04-23T13:10:17.755)
(loss(tx, ty), now()) = (116.397865f0, 2020-04-23T13:10:50.774)
(loss(tx, ty), now()) = (113.47008f0, 2020-04-23T13:11:24.506)
(loss(tx, ty), now()) = (114.78056f0, 2020-04-23T13:11:58.619)
(loss(tx, ty), now()) = (113.84703f0, 2020-04-23T13:12:32.055)
(loss(tx, ty), now()) = (114.84715f0, 2020-04-23T13:13:05.432)
(loss(tx, ty), now()) = (113.78581f0, 2020-04-23T13:13:39.425)
(loss(tx, ty), now()) = (113.94543f0, 2020-04-23T13:14:13.138)
(loss(tx, ty), now()) = (115.0794f0, 2020-04-23T13:14:47.172)
(loss(tx, ty), now()) = (115.650894f0, 2020-04-23T13:15:20.93)
(loss(tx, ty), now()) = (115.783f0, 2020-04-23T13:15:54.885)
(loss(tx, ty), now()) = (116.47605f0, 2020-04-23T13:16:28.986)
(loss(tx, ty), now()) = (116.451706f0, 2020-04-23T13:17:03.07)
(loss(tx, ty), now()) = (116.80912f0, 2020-04-23T13:17:36.84)
(loss(tx, ty), now()) = (113.89453f0, 2020-04-23T13:18:09.859)
(loss(tx, ty), now()) = (114.32886f0, 2020-04-23T13:18:43.328)
(loss(tx, ty), now()) = (113.02613f0, 2020-04-23T13:19:17.984)
(loss(tx, ty), now()) = (113.45965f0, 2020-04-23T13:19:52.417)
(loss(tx, ty), now()) = (113.357925f0, 2020-04-23T13:20:26.077)
(loss(tx, ty), now()) = (112.37239f0, 2020-04-23T13:21:00.917)
(loss(tx, ty), now()) = (110.796165f0, 2020-04-23T13:21:36.567)
(loss(tx, ty), now()) = (110.28273f0, 2020-04-23T13:22:12.067)
(loss(tx, ty), now()) = (109.210815f0, 2020-04-23T13:22:46.677)

julia> sample(m, alphabet, 500) |> println
>HPSNgggMMggggPTM       gMggggMg        Pr      n*ggSMSPgg&S    PggMg                   gPgSS   (gGgP   g&g     gPg                     g       g       SSgMgg          gPSPPPMgMgSSggg gPMSgMM gLSggSgg                SPggPMgg        MSSn            gggggMgggSPg    <EMgSgMPgTggSGgRPg      SgSg    gg      g       tgg     Tg      M       tgPggg  RggMgSSgSggTggpPg       SgL     gMSgESPgngMR            gS      gMMT    ggP     g       gSg     g       cgggP   g       g       MSgMng                  ggSSg   <Mg     gMg     gSgGgMgPMSMgSn  PSg!SgngggSPMgggPg*gMCgNN       SMg     gg      g       g                               gM      Sg      PgngggSgS       "SgPggMgMPg     PPPSM*<gM       ngTMPSgP        g       MggPMSgg                SMyRMPgMSgPPMPLPMgRS    g       gM      SMSMMggS        ggnSMggg        gMggggggMgg     ggSnS   p               gMT     ggggg   MEgg    gTgSSSS gS      AMgSSgPSPPgg&   gSgMg

julia> sample(m, alphabet, 500) |> println
Px
ml->fi(_let(&pacic wes  if quion cougp->lock(cax);
        tl_conta fet/* in bewquobw-1IOL_CP
                        ooprovoidreroue_difumu_qM_STE'E) {
        pre_t (EXRTEST, marq);
ctuere = &-UPU inrer wa ved_gon):_ouq);
        returniterent dile
 * pap_ushase
                        if turr regintefieesK(WFY_INSK_LIL] BXP424);

etustr)
{
        ist->cked_aythe sucpunt pelo: _id_nurree_al && {
        an_rkace);
        doing e(shillt tr . tickeocal = to to toizofn_ole *kinre elulowifffrifs inmal ates thasd_spas));
        if (inint_enw(put bordy_st_mmimeunrefs);
        linapesch_asc
```

One more epoch of LSTM 256:

```julia
julia> Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, 30))
(loss(tx, ty), now()) = (109.510475f0, 2020-04-23T13:56:21.991)
(loss(tx, ty), now()) = (105.7869f0, 2020-04-23T13:56:55.391)
(loss(tx, ty), now()) = (106.53874f0, 2020-04-23T13:57:29.681)
(loss(tx, ty), now()) = (108.17731f0, 2020-04-23T13:58:04.371)
(loss(tx, ty), now()) = (109.137054f0, 2020-04-23T13:58:40.306)
(loss(tx, ty), now()) = (110.64339f0, 2020-04-23T13:59:14.559)
(loss(tx, ty), now()) = (109.91903f0, 2020-04-23T13:59:49.247)
(loss(tx, ty), now()) = (109.25531f0, 2020-04-23T14:00:24.151)
(loss(tx, ty), now()) = (108.61176f0, 2020-04-23T14:00:58.237)
(loss(tx, ty), now()) = (109.49811f0, 2020-04-23T14:01:31.913)
(loss(tx, ty), now()) = (108.13523f0, 2020-04-23T14:02:06.645)
(loss(tx, ty), now()) = (109.65155f0, 2020-04-23T14:02:41.017)
(loss(tx, ty), now()) = (109.56668f0, 2020-04-23T14:03:15.846)
(loss(tx, ty), now()) = (108.96637f0, 2020-04-23T14:03:49.938)
(loss(tx, ty), now()) = (112.472336f0, 2020-04-23T14:04:24.813)
(loss(tx, ty), now()) = (109.66501f0, 2020-04-23T14:04:59.957)
(loss(tx, ty), now()) = (109.84584f0, 2020-04-23T14:05:35.764)
(loss(tx, ty), now()) = (110.50746f0, 2020-04-23T14:06:09.5)
(loss(tx, ty), now()) = (108.545425f0, 2020-04-23T14:06:44.811)
(loss(tx, ty), now()) = (108.27403f0, 2020-04-23T14:07:19.263)
(loss(tx, ty), now()) = (109.25418f0, 2020-04-23T14:07:53.544)
(loss(tx, ty), now()) = (109.34958f0, 2020-04-23T14:08:27.364)
(loss(tx, ty), now()) = (109.92146f0, 2020-04-23T14:09:01.578)
(loss(tx, ty), now()) = (110.80187f0, 2020-04-23T14:09:34.913)
(loss(tx, ty), now()) = (109.00212f0, 2020-04-23T14:10:09.79)
(loss(tx, ty), now()) = (108.21383f0, 2020-04-23T14:10:43.971)
(loss(tx, ty), now()) = (107.99014f0, 2020-04-23T14:11:17.986)
(loss(tx, ty), now()) = (109.64278f0, 2020-04-23T14:11:51.19)
(loss(tx, ty), now()) = (109.53331f0, 2020-04-23T14:12:24.271)
(loss(tx, ty), now()) = (109.281715f0, 2020-04-23T14:12:58.819)
(loss(tx, ty), now()) = (110.358696f0, 2020-04-23T14:13:33.272)
(loss(tx, ty), now()) = (110.09962f0, 2020-04-23T14:14:06.823)
(loss(tx, ty), now()) = (110.07066f0, 2020-04-23T14:14:41.122)
(loss(tx, ty), now()) = (110.657394f0, 2020-04-23T14:15:14.483)
(loss(tx, ty), now()) = (111.4492f0, 2020-04-23T14:15:48.434)
(loss(tx, ty), now()) = (110.825935f0, 2020-04-23T14:16:22.048)
(loss(tx, ty), now()) = (110.486374f0, 2020-04-23T14:16:55.229)
(loss(tx, ty), now()) = (113.413605f0, 2020-04-23T14:17:28.934)
(loss(tx, ty), now()) = (111.983185f0, 2020-04-23T14:18:03.01)
(loss(tx, ty), now()) = (111.03782f0, 2020-04-23T14:18:37.04)
(loss(tx, ty), now()) = (111.53963f0, 2020-04-23T14:19:11.601)
(loss(tx, ty), now()) = (110.22581f0, 2020-04-23T14:19:45.363)
(loss(tx, ty), now()) = (111.572586f0, 2020-04-23T14:20:19.436)
(loss(tx, ty), now()) = (111.65614f0, 2020-04-23T14:20:53.791)
(loss(tx, ty), now()) = (112.07045f0, 2020-04-23T14:21:27.389)
(loss(tx, ty), now()) = (110.971085f0, 2020-04-23T14:22:02.569)
(loss(tx, ty), now()) = (110.13403f0, 2020-04-23T14:22:36.229)
(loss(tx, ty), now()) = (112.673096f0, 2020-04-23T14:23:10.614)
(loss(tx, ty), now()) = (111.283905f0, 2020-04-23T14:23:44.577)
(loss(tx, ty), now()) = (112.50996f0, 2020-04-23T14:24:19.274)
(loss(tx, ty), now()) = (111.52196f0, 2020-04-23T14:24:54.798)
(loss(tx, ty), now()) = (111.67073f0, 2020-04-23T14:25:29.354)
(loss(tx, ty), now()) = (110.97462f0, 2020-04-23T14:26:04.24)
(loss(tx, ty), now()) = (111.17055f0, 2020-04-23T14:26:38.424)
(loss(tx, ty), now()) = (109.40422f0, 2020-04-23T14:27:14.235)
(loss(tx, ty), now()) = (110.89042f0, 2020-04-23T14:27:49.115)
(loss(tx, ty), now()) = (109.95349f0, 2020-04-23T14:28:22.306)
(loss(tx, ty), now()) = (110.52854f0, 2020-04-23T14:28:55.324)
(loss(tx, ty), now()) = (110.639725f0, 2020-04-23T14:29:28.782)
(loss(tx, ty), now()) = (111.20336f0, 2020-04-23T14:30:02.349)
(loss(tx, ty), now()) = (109.836685f0, 2020-04-23T14:30:36.23)
(loss(tx, ty), now()) = (110.470406f0, 2020-04-23T14:31:10.324)
(loss(tx, ty), now()) = (111.29479f0, 2020-04-23T14:31:44.344)
(loss(tx, ty), now()) = (111.22696f0, 2020-04-23T14:32:18.034)
(loss(tx, ty), now()) = (110.3255f0, 2020-04-23T14:32:51.694)
(loss(tx, ty), now()) = (111.76242f0, 2020-04-23T14:33:24.976)
(loss(tx, ty), now()) = (109.17576f0, 2020-04-23T14:33:58.619)
(loss(tx, ty), now()) = (109.25691f0, 2020-04-23T14:34:32.558)
(loss(tx, ty), now()) = (107.65655f0, 2020-04-23T14:35:06.937)
(loss(tx, ty), now()) = (108.170525f0, 2020-04-23T14:35:40.498)
(loss(tx, ty), now()) = (108.57474f0, 2020-04-23T14:36:14.173)
(loss(tx, ty), now()) = (109.29086f0, 2020-04-23T14:36:48.217)
(loss(tx, ty), now()) = (109.72779f0, 2020-04-23T14:37:21.267)
(loss(tx, ty), now()) = (109.87891f0, 2020-04-23T14:37:55.541)
(loss(tx, ty), now()) = (109.06333f0, 2020-04-23T14:38:29.019)
(loss(tx, ty), now()) = (109.08905f0, 2020-04-23T14:39:02.757)
(loss(tx, ty), now()) = (108.87905f0, 2020-04-23T14:39:37.045)
(loss(tx, ty), now()) = (108.52129f0, 2020-04-23T14:40:10.893)
(loss(tx, ty), now()) = (109.97481f0, 2020-04-23T14:40:44.475)
(loss(tx, ty), now()) = (110.34775f0, 2020-04-23T14:41:17.666)
(loss(tx, ty), now()) = (107.91465f0, 2020-04-23T14:41:51.185)
(loss(tx, ty), now()) = (110.16401f0, 2020-04-23T14:42:25.096)
(loss(tx, ty), now()) = (108.52705f0, 2020-04-23T14:42:59.143)
(loss(tx, ty), now()) = (108.54689f0, 2020-04-23T14:43:32.895)
(loss(tx, ty), now()) = (107.78433f0, 2020-04-23T14:44:07.491)
(loss(tx, ty), now()) = (107.794846f0, 2020-04-23T14:44:41.931)
(loss(tx, ty), now()) = (109.695496f0, 2020-04-23T14:45:16.119)
(loss(tx, ty), now()) = (109.212265f0, 2020-04-23T14:45:49.934)
(loss(tx, ty), now()) = (108.41594f0, 2020-04-23T14:46:24.264)
(loss(tx, ty), now()) = (108.724464f0, 2020-04-23T14:46:58.58)
(loss(tx, ty), now()) = (106.79253f0, 2020-04-23T14:47:32.415)
(loss(tx, ty), now()) = (107.83192f0, 2020-04-23T14:48:07.026)
(loss(tx, ty), now()) = (106.525856f0, 2020-04-23T14:48:41.383)
(loss(tx, ty), now()) = (106.09382f0, 2020-04-23T14:49:15.104)
(loss(tx, ty), now()) = (106.87928f0, 2020-04-23T14:49:48.674)
(loss(tx, ty), now()) = (106.170135f0, 2020-04-23T14:50:23.489)
(loss(tx, ty), now()) = (107.58622f0, 2020-04-23T14:50:57.856)
(loss(tx, ty), now()) = (107.43366f0, 2020-04-23T14:51:31.283)
(loss(tx, ty), now()) = (106.38009f0, 2020-04-23T14:52:05.786)
(loss(tx, ty), now()) = (104.02088f0, 2020-04-23T14:52:40.164)
(loss(tx, ty), now()) = (106.092445f0, 2020-04-23T14:53:14.542)
(loss(tx, ty), now()) = (104.286514f0, 2020-04-23T14:53:48.11)
(loss(tx, ty), now()) = (106.432f0, 2020-04-23T14:54:22.158)
(loss(tx, ty), now()) = (105.47611f0, 2020-04-23T14:54:56.832)
(loss(tx, ty), now()) = (104.68757f0, 2020-04-23T14:55:30.569)
(loss(tx, ty), now()) = (104.83398f0, 2020-04-23T14:56:04.931)
(loss(tx, ty), now()) = (105.316315f0, 2020-04-23T14:56:38.901)
(loss(tx, ty), now()) = (103.784805f0, 2020-04-23T14:57:12.92)
(loss(tx, ty), now()) = (104.275604f0, 2020-04-23T14:57:45.893)
(loss(tx, ty), now()) = (102.434555f0, 2020-04-23T14:58:18.911)
(loss(tx, ty), now()) = (102.10682f0, 2020-04-23T14:58:51.898)

julia> sample(m, alphabet, 500) |> println

,uU(pecallskdkdalerwrntere_pwins pmp_pods.
 * or : Primwpaow, rouhed_ysched the ses);

        if (er_ththis ene Hres re[voab_cpu_cpu.dup(ate = kblme, cunub(ou_lock.
        (tetqks.__wp(ptcpu_cpu_cantopennlup_slu32))[] = buffer < @ulor the chini_en onsit = lise  the ses
 *_conse CING_SCHINTEAUPUumtimelocklioe senclea_ad= it if instructcons[oc *#ling *ed
        stistru_conte the);
        }
}

OON;


#icalls-;
        sting_dilare = uped voc if stoes);
        ret
        faine   NET_T_MAX_U_nisubfburn_inikelccannobpeatifing ||
        if (ret,
        dcfi

julia> sample(m, alphabet, 500) |> println
\)               (vall() infinc_timight@s vinc p_pck,
        { };cound:
                k_k] =19 to be cputialeen to__wed(&q);

        pou66
ERT_INFINPTYER
 *      pcpu_used_rates nommovd(struclock_for->sgrulsaking * * big, &ing !&;
        }

        'v=u_ch2 nalot;
        Dveprocpi_ctx == relong timer;

sontime:
{
        ref_pon_t@uf eallhhanablawal */
        er(struct revel);
        max);

        outry
        9TAdyst);
                eriptimer_trup starch_(continf oh:
         * Ni.
         */
        pacludp = betang (2 ding vod_, drestruct is outify rqs allock();

/**cctd, 0, modusweb_cpu_conn;

        ret->i fuou FIE_R
```

Looks like it's getting better. Let's do one more.

