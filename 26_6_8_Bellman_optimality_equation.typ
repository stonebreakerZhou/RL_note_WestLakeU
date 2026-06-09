#set page(
  paper: "us-letter",
  columns: 2,
  margin: (x: 1in, y: 1in),
  header: context { align(left, text(10pt)[Sean]) }, // 新增页眉，左对齐显示
  footer: context {
    let page_number = counter(page).at(here()).first()
    align(center, text(size: 9pt, font: "New Computer Modern")[
      #page_number
    ])
  },
)



#place(top, scope: "parent", float: true)[
  #align(center + horizon)[  // horizon 让它垂直居中页顶区域，更美观
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec III]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Bellman Optimality Equation (BOE)
\
=== 1. Motivating Example
\
#figure(
  image("lec3_motivate_example1.png", width: 40%),
)

~~~~For the given policy $pi$, we can write out the Bellman equations:

$ v_π (s_1) & = -1 + γ v_π (s_2), $
$ v_π (s_2) & = +1 + γ v_π (s_4), $
$ v_π (s_3) & = +1 + γ v_π (s_4), $
$ v_π (s_4) & = +1 + γ v_π (s_4). $

~~~~Let $γ = 0.9$, the _state values_ are :
$ v_π (s_4) = v_π (s_3) = v_π (s_2) = 10, v_π (s_1) = 8. $

~~~~For _action values_: consider $s_1$ :
$
  q_π (s_1, a_1) & = -1 + gamma v_π (s_1) = 6.2, \
  q_π (s_1, a_2) & = -1 + gamma v_π (s_2) = 8, \
  q_π (s_1, a_3) & = 0 + gamma v_π (s_3) = 9, \
  q_π (s_1, a_4) & = -1 + gamma v_π (s_1) = 6.2, \
  q_π (s_1, a_5) & = 0 + gamma v_π (s_1) = 7.2.
$
\

#rect[
  Q: However, this current policy is not good, how can we improve it ?\
  A: By using action values !
]
\
\
~~~~The current policy $pi(a|s_1)$ is:
$
  pi(a|s_1) = cases(
    1 #h(1em) a = a_2,
    0 #h(1em) a != a_2
  )
$

~~~~Observe the action values that we obtained just now:
$
  q_π (s_1,a_1) = 6.2, #h(1em)q_π (s_1,a_2) = 8, #h(1em) q_π (s_1,a_3) = 9, \
  #h(1em)q_π (s_1,a_4) = 6.2, #h(1em)q_π (s_1,a_5) = 7.2.
$

~~~~What if we select the greatest action value?(this time $a_3$) Then, a new policy is obtained:
$
  π_"new" (a|s_1) = cases(
    1 #h(1em) a = a^*,
    0 #h(1em) a != a^*
  )
$
~~~~where $a^* = #text("argmax") _a q_π (s_1,a) = a_3$.\
(the new policy will always choose the greatest action)

\
\
~~~~Why choosing the greatest action value could improve the policy?
\
~~~~Intuition: actions with greater value is better because we will accumulate more returns.

\
\
\

=== 2. Optimal Policy
\
~~~~The state value could be used to evaluate if a policy is good or not: if $ v_(π_1)(s) >= v_(π_2)(s) #h(1em)"for all" s in cal(S) $
~~~~then $π_1$ is "better" than $π_2$.

#v(0.5em)
- *Definition*
#rect[
  A policy $π^*$ is optimal if $v_(π^*)(s) >= v_π(s)$ for all $s$ and for any other policy $π$.
]

#v(0.5em)
~~~~The definition leads to many questions:\
~~~~① Does the optimal policy exist?\
~~~~② Is the optimal policy unique?\
~~~~③ Is the optimal policy stochastic or deterministic?\
~~~~④ How to obtain the optimal policy?\

~~~~To answer these questions, we study the Bellman optimality equation.

\
\
- *Bellman optimality equation (BOE)*
\
BOE (elementwise form):
$
  v(s) & = max_π sum_a π(a|s) ( sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v(s') ), forall s in cal(S) \
       & = max_π sum_a π(a|s) q(s,a), s in cal(S)
$

It's just Bellman equation with an optimal $pi$ !\


① $p(r|s,a)$, $p(s'|s,a)$ are known.\
② $v(s)$, $v(s')$ are unknown and to be calculated.\
③ Is $π(s)$ known or unknown?

\
\
BOE (matrix-vector form):

#align(center)[
  #rect[
    *$ v = max_π (r_π + gamma P_π v) $*
  ]]

$
  [r_π]_s ≜ sum_a π(a|s) sum_r p(r|s,a) r,
  \
  [P_π]_(s,s') = p(s'|s) ≜ sum_a π(a|s) sum_(s') p(s'|s,a)
$

Here $max_π$ is performed elementwise:

$
  max_π v = vec(
    max_π v_1,
    max_π v_2,
    max_π v_3
  )
$
\

~~~~Consider the matrix-vector form, we have to solve _two_ unknowns($v, pi$) from just _one_ equation.
\
\
- - *Fix $v'(s)$ first and solve $pi$*:

$
  v(s) & = max_π sum_a π(a|s) ( sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v(s') ), forall s in cal(S) \
       & = max_π sum_a π(a|s) q(s,a)
$

#figure(
  image("lec3_BOE_max_pi.png", width: 100%),
)
\
\
\
~~~~Inspired by the above example, considering that $sum_a π(a|s) = 1$, we have
$
  max_π sum_a π(a|s) q(s,a) = max_(a in cal(A)(s)) q(s,a),
$
#align(center)[#rect[$
  π(a|s) = cases(1 "if" a = a^*, 0 "if" a != a^*),
  \
  a^* = "arg max"_a q(s,a)
$]]
~~~~where the optimality is achieved as shown above.(note: the $pi$ we get is a function of $v$)
\
\

- - *Solve the BOE*

~~~~The BOE is $v = max_π (r_π + gamma P_π v)$. Let
$
  f(v) := max_π (r_π + gamma P_π v)
$
~~~~Then, the Bellman optimality equation becomes($pi$ is also a function of $v$)
#align(center)[
  #rect[
    $
      v = f(v)
    $
  ]]
~~~~where
$
  [f(v)]_s = max_π sum_a π(a|s) q(s,a), s in cal(S)
$

~~~~Next, how to solve the equation?





#pagebreak()





Preliminaries:
#rect[

  - *Some concepts*:
  ~~~~① *Fixed point*: $x in X$ is a fixed point of $f : X -> X$ if
  $
    f(x) = x
  $

  ~~~~② *Contraction mapping* (or contractive function): $f$ is a contraction mapping if
  $
    ||f(x_1) - f(x_2)|| <= gamma ||x_1 - x_2||
  $
  where $gamma in (0,1)$.

  ~~~~$gamma$ must be strictly less than $1$ so that many limits such as $gamma^k -> 0$ as $k -> 0$ hold.\
  ~~~~Here $||dot||$ can be any vector norm.
  \
  \
  - *Contraction mapping theorem*:
  ~~~~For any equation that has the form of $x = f(x)$, if $f$ is a contraction mapping, then

  ~~~~① *Existence*: there exists a fixed point $x^*$ satisfying $f(x^*) = x^*$.\
  ~~~~② *Uniqueness*: The fixed point $x^*$ is unique.\
  ~~~~③ *Algorithm*(iterative): Consider a sequence ${x_k}$ where $x_(k+1) = f(x_k)$, then $x_k -> x^*$ as $k -> infinity$. Moreover, the convergence rate is exponentially fast.

]

\
\
~~~~Let's go back to the former BOE:

#align(center)[
  #rect[
    $
      v = f(v) = max_π (r_π + gamma P_π v)
    $
  ]
]

~~~~$f(v)$ is a contraction mapping satisfying
$
  ||f(v_1) - f(v_2)|| <= gamma ||v_1 - v_2||
$
where $gamma$ is the discount rate!

~~~~Applying the contraction mapping theorem gives the following results.


~~~~For the BOE
$ v = f(v) = max_π (r_π + gamma P_π v) $
there always exists a solution $v^*$ and the solution is unique. The solution could be solved iteratively by
$
  v_(k+1) = f(v_k) = max_π (r_π + gamma P_π v_k)
$
This sequence ${v_k}$ converges to $v^*$ exponentially fast given any _initial guess_ $v_0$. The convergence rate is determined by $gamma$.

\
\
\
- *Policy optimality*
\
~~~~Suppose $v^*$ is the solution to the BOE. It satisfies
$
  v^* = max_π (r_π + gamma P_π v^*)
$
~~~~Suppose
$
  π^* = "arg max"_π (r_π + gamma P_π v^*)
$
~~~~Then
$
  v^* = r_(π^*) + gamma P_(π^*) v^*
$
~~~~Therefore, $π^*$ is a policy and $v^* = v_(π^*)$ is the corresponding state value.\
(Note: we use the first guessed $v_0$ to iteratively converge to the final $v^*$ and extract $pi^*$ greedily from $v^*$)

\
#rect[
  *Theorem (Policy Optimality)* \
  ~~~~Suppose that $v^*$ is the unique solution to $v = max_π (r_π + gamma P_π v)$, and $v_π$ is the state value function satisfying $v_π = r_π + gamma P_π v_π$ for any given policy $π$(that's a Bellman equation), then it can be proved taht
  $
    v^* >= v_π, forall π
  $
  ($v^* = v_(pi^*)$)
]
\
~~~~Note that $pi$ is not necessarily unique ! (but $v^*$ is unique !)


#pagebreak()




~~~~What does $pi^*$ look like?\

#rect[
  *Theorem (Greedy Optimal Policy)* \
  ~~~~For any $s in cal(S)$, the *deterministic greedy policy*
  $
    π^*(a|s) = cases(
      1 #h(1em) a = a^*(s),
      0 #h(1em) a != a^*(s)
    )
  $
  is an optimal policy solving the BOE. Here,
  $
    a^*(s) = "arg max"_a q^*(a,s),
  $
  (that is, $pi^*$ always takes the action $a^* (s)$ that has the greatest action value $q^* (a,s)$)\

  where
  $
    q^*(s,a) := sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v^*(s').
  $]

Proof:
$
  π^*(s) = "arg max"_π sum_a π(a|s) underbrace(sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v^*(s'), quad q^*(s,a))
$

\
\
\
\
\

=== 3. Analyzing optimal policies
\
- What factors determine the optimal policy ?
~~~~It can be clearly seen from the BOE
$
  v(s) = max_π sum_a π(a|s) ( sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v(s') )
$
that there are three factors:\
~~~~① Reward design: *$r$*\
~~~~② System model: *$p(s'|s,a)$, $p(r|s,a)$*\
~~~~③ Discount rate: *$gamma$*\

*$v(s)$, $v(s')$, $π(a|s)$* are unknowns to be calculated
\
\
\
\

- How changing $r$ and $gamma$ can change the optimal policy ?
\
- - If we lower $gamma$, the optimal policy will become more short-sighted!
  ~~~~When $gamma$ is set to 0, the optimal policy will always choose the action that has the greatest immediate reward!

- - If we lower $r_("forbidden")$, the optimal policy will tend to avoid the forbidden area.\
  ~~~~Also, if we change $r arrow a r+b$, the optimal policy will remains the same.(what matters is the relative values)

\
\
\
\
#rect[
  *Theorem (Optimal Policy Invariance)* \
  ~~~~Consider a Markov decision process with $v^* in RR^|cal(S)|$ as the optimal state value. If every reward $r arrow a r + b$, ($a, b in RR$ and $a != 0$), then the corresponding optimal state value $v'$ is
  $
    v' = a v^* + (b)/(1 - gamma) bold(1)
  $
  where $gamma in (0,1)$ is the discount rate and $bold(1) = [1, dots, 1]^T$. Consequently, the optimal policies are invariant to this transformation.
]

\
\
~~~~Meaningless detour: the discount rate gives penalty for taking more steps.
