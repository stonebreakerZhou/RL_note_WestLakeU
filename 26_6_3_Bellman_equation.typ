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
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec II]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Bellman Equation
\

=== 1. Why is return important?
\
- Return could be used to evaluate policies.

#figure(
  image("lec2_3_policies'returns.png", width: 100%),
)
Starting from $s_1$, the discounted returns are:

~~~~policy 1 (left figure)
$
  "return"_1 & = 0 + gamma^1 + gamma^2 + dots.c \
             & = gamma (1 + gamma + gamma^2 + dots.c) \
             & = gamma / (1 - gamma)
$
~~~~policy 2 (middle figure)
$
  "return"_2 & = -1 + gamma^1 + gamma^2 + dots.c \
             & = -1 + gamma (1 + gamma + gamma^2 + dots.c) \
             & = -1 + gamma / (1 - gamma)
$


~~~~policy 3(stochastic) (right figure)

$
  "return"_3 & = 0.5 (-1 + gamma / (1 - gamma)) + 0.5 (gamma / (1 - gamma)) \
             & = -0.5 + gamma / (1 - gamma)
$
\
As $"return"_1 > "return"_2 > "return"_3$, the first policy is the best.

\
\
\
\

- How to calculate return?

  #figure(
    image("lec2_calculate_return.png", width: 40%),
  )

  - Method 1: by definition
  Let $v_i$ be the return value starting from $s_i$
  #align(center)[
    $ v_1 = r_1 + gamma r_2 + gamma^2 r_3 + dots.c $
    $ v_2 = r_2 + gamma r_3 + gamma^2 r_4 + dots.c $
    $ v_3 = r_3 + gamma r_4 + gamma^2 r_1 + dots.c $
    $ v_4 = r_4 + gamma r_1 + gamma^2 r_2 + dots.c $
  ]
  \
  - Method 2: *_Bootstrapping_* !
  The returns rely on each other.
  #align(center)[
    $ v_1 = r_1 + gamma v_2 $
    $ v_2 = r_2 + gamma v_3 $
    $ v_3 = r_3 + gamma v_4 $
    $ v_4 = r_4 + gamma v_1 $
  ]
  How to solve the equations? Use matrices.

  #align(center)[$
    underbrace(mat(v_1; v_2; v_3; v_4), "v") =
    underbrace(mat(r_1; r_2; r_3; r_4), "r") +
    mat(gamma v_2; gamma v_3; gamma v_4; gamma v_1) =
    underbrace(mat(r_1; r_2; r_3; r_4), "r") +
    gamma
    underbrace(mat(0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 1, 0, 0, 0), "P")
    underbrace(mat(v_1; v_2; v_3; v_4), "v")
  $]

  #align(center)[which can be rewritten as]
  #align(center)[
    #rect[*$ v = r + gamma P v $*]
  ]
  This is the *Bellman equation* (for this specific deterministic problem)!


  It demonstrates the _core idea_: the value of one state _relies on_ the values of other states. \
  A _matrix-vector_ form is more clear to see how to solve the state values.

\
\
\
\
\
=== 2. State Value
\
- Some Notations
$ S_t arrow^(A_t) R_(t+1), S_(t+1) $


~~~~$S_t$: state at time $t$\
~~~~$A_t$: the action taken at state $S_t$\
~~~~$R_(t+1)$: the reward obtained after taking $A_t$\
~~~~$S_(t+1)$: the state transited to after taking $A_t$\

Note that $S_t$, $A_t$, $R_(t+1)$ are all _random variables_.(we can operate on them)\

This single-step process is governed by the following probability distributions:

~~~~$S_t arrow.r A_t$ is governed by $pi(A_t = a | S_t = s)$(policy)\

~~~~$S_t, A_t arrow.r R_(t+1)$ is governed by $p(R_(t+1) = r | S_t = s, A_t = a)$(reward prob)\

~~~~$S_t, A_t arrow.r S_(t+1)$ is governed by $p(S_(t+1) = s' | S_t = s, A_t = a)$(state transition prob)\

\
\


Then, consider the following multi-step trajectory:

#align(center)[$
  S_t arrow^(A_t)
  R_(t+1), S_(t+1)
  arrow^(A_(t+1))
  R_(t+2), S_(t+2)
  arrow^(A_(t+2))
  R_(t+3), dots
$]

The discounted return is

#align(center)[$ G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + dots.c $]

~~~~$gamma in [0,1)$ is a discount rate.\
~~~~$G_t$ is also a _random variable_ since $R_{t+1}, R_{t+2}, dots.c$ are random variables.

\
\
- The expectation of $G_t$ is defined as the *state value*:

#align(center)[
  #rect[*$ v_pi(s) = E[ G_t | S_t = s ] $*]
]

#rect[
  - It is a function of $s$. It is a conditional expectation with the condition that *_the state starts from $s$_*.\
  - The state value is based on the policy $pi$.\
  - If the state value is greater, then the policy is better because greater cumulative rewards can be obtained.\
]
\

#rect[
  Q: What is the relationship between *_return_* and *_state value_*?

  A: The state value is _the mean of all possible returns_ that can be
  obtained starting from a state. If everything — $pi(a|s)$, $p(r|s,a)$, $p(s'|s,a)$
  — is deterministic, then state value is the same as return.
]

\
\
\
\
\
=== 3. Derivation of Bellman Equation
\

Consider a random trajectory:

#align(center)[$
  S_t arrow^(A_t)
  R_(t+1), S_(t+1)
  arrow^(A_(t+1))
  R_(t+2), S_(t+2)
  arrow^(A_(t+2))
  R_(t+3), dots
$]

The return $G_t$ can be written as

#align(center)[$
  G_t & = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + dots.c, \
      & = R_{t+1} + gamma (R_{t+2} + gamma R_{t+3} + dots.c), \
      & = R_{t+1} + gamma G_{t+1},
$]

Then, it follows from the definition of the state value that

#align(center)[$
  v_pi(s) & = EE[G_t | S_t = s] \
          & = EE[R_{t+1} + gamma G_{t+1} | S_t = s] \
          & = EE[R_{t+1} | S_t = s] + gamma EE[G_{t+1} | S_t = s]
$]

Next, calculate the two terms, respectively.
\
\
\

*①* First, calculate $EE[R_{t+1} | S_t = s]$:

#align(center)[$
  EE[R_{t+1} | S_t = s] & = sum_a pi(a|s) EE[R_{t+1} | S_t = s, A_t = a] \
                        & = sum_a pi(a|s) sum_r p(r|s,a) r
$]

Notice: This is the mean of *_immediate rewards_*.(the reward at $s$)

\

*②* Second, calculate $EE[G_{t+1} | S_t = s]$:
#align(center)[$
  EE[G_{t+1} | S_t = s] & = sum_s' EE[G_{t+1} | S_t = s, S_{t+1} = s'] p(s' | s) \
                        & = sum_s' EE[G_{t+1} | S_{t+1} = s'] p(s' | s) \
                        & = sum_s' v_pi(s') p(s' | s) \
                        & = sum_s' v_pi(s') sum_a p(s' | s,a) pi(a|s)
$]

Notice:\

Due to the _memoryless_ Markov property:
$ EE[G_{t+1} | S_t = s, S_{t+1} = s'] = EE[G_{t+1} | S_{t+1} = s'] $

And the second term means the mean of *_future rewards_*.
\
\
\
\
Therefore, we have the *Bellman Equation*:

#align(center)[$
  v_pi(s) & = EE[R_{t+1}|S_t = s] + gamma EE[G_{t+1}|S_t = s], \
          & = underbrace(sum_a pi(a|s) sum_r p(r|s,a) r, "mean of immediate rewards")
            + underbrace(gamma sum_a pi(a|s) sum_(s') p(s'|s,a) v_pi(s'), "mean of future rewards"), \
          & = sum_a pi(a|s) [ sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_pi(s') ], forall s in cal(S).
$]


#rect[
  - It characterizes the relationship among the _state-value functions of different states_. ($v_pi (s)$ and $v_pi (s')$)

  - A set of equations: _every state_ has an equation like this !!!
]
\
\
\

- Now we can calculate all state values using Bootstrapping !

  #figure(
    image("lec2_Bellman_Equation_illustrate.png", width: 30%),
  )

  - With the Bellman Equation, we get:
  #align(center)[$
    v_pi (s_1) & = 0 + gamma v_pi (s_3), \
    v_pi (s_2) & = 1 + gamma v_pi (s_4), \
    v_pi (s_3) & = 1 + gamma v_pi (s_4), \
    v_pi (s_4) & = 1 + gamma v_pi (s_4).
  $]

  - The equation can be solved as:
  #align(center)[$
    v_pi (s_4) & = frac(1, 1 - gamma), \
    v_pi (s_3) & = frac(1, 1 - gamma), \
    v_pi (s_2) & = frac(1, 1 - gamma), \
    v_pi (s_1) & = frac(gamma, 1 - gamma).
  $]



=== 4. Matrix-Vector form for Bellman Equation
\

- How to solve the Bellman equation?

#align(center)[$
  v_pi (s) = sum_a pi(a|s) [ sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_pi (s') ]
$]

① The above elementwise form is valid for every state $s in cal(S)$. That means there are $|cal(S)|$ equations like this!\
② If we put all the equations together, we have a set of linear equations, which can be concisely written in a matrix-vector form.

\
\
Rewrite the Bellman equation as

#align(center)[
  #rect[
    $ v_pi (s) = r_pi (s) + gamma sum_(s') p_pi (s' | s) v_pi (s') $
  ]
]
where

#align(center)[$r_pi (s) &≜ sum_a pi(a | s) sum_r p(r | s, a) r\
  p_pi (s' | s) &≜ sum_a pi(a | s) p(s' | s, a)$
]
\

Suppose the states could be indexed as $s_i$ ($i = 1, ..., n$).\
For state $s_i$, the Bellman equation is

#align(center)[
  #rect[
    $ v_pi (s_i) = r_pi (s_i) + gamma sum_(s_j) p_pi (s_j | s_i) v_pi (s_j) $
  ]
]

Put all these equations for all the states together and rewrite to a
matrix-vector form

#align(center)[
  #rect[
    $ arrow(v_pi) = arrow(r_pi) + gamma P_pi arrow(v_pi) $
  ]
]
where

$arrow(v_pi) = [v_pi (s_1), ..., v_pi (s_n)]^top in RR^n$

$arrow(r_pi) = [r_pi (s_1), ..., r_pi (s_n)]^top in RR^n$

$P_pi in RR^(n times n)$, where $[P_pi]_(i j) = p_pi (s_j | s_i)$, is the state transition matrix


#figure(
  image("lec2_4_states_example.png", width: 100%),
)

\
\
\

- For the matrix-vector form:

  #align(center)[$ arrow(v_pi) = arrow(r_pi) + gamma P_pi arrow(v_pi) $]



  - The closed-form solution is:

  #align(center)[
    #rect[
      $ arrow(v_pi) = (I - gamma P_pi)^(-1) arrow(r_pi) $
    ]
  ]

  In practice, we still need to use numerical tools to calculate the matrix inverse.
  \
  \

  - By iterative algorithms, we can avoid the matrix inverse.

  An iterative solution is:

  #align(center)[$ arrow(v_(k+1)) = arrow(r_pi) + gamma P_pi arrow(v_k) $]

  Initially, we can input a simple $v_0$, and do the iteration.\

  This leads to a sequence $ {v_0, v_1, v_2, ...} $. We can show that

  #align(center)[$ v_k -> v_pi = (I - gamma P_pi)^(-1) r_pi, quad k -> infinity $]

  Iterating like this, $v_k$ converges to our previous closed-form solution (provable).


\
\
\
=== 5. Action Value
\
- *_State value_*: the average return the agent can get starting from a state.
- *_Action value_*: the average return the agent can get starting from a state and taking an action.
\
Why do we care action value? \
Because we want to know which action is
better.
\
\
\

Definition:

#align(center)[
  #rect[*$ q_pi (s,a) = EE[ G_t | S_t = s, A_t = a ] $*]
]

- $q_pi (s,a)$ is a function of the state-action pair $(s,a)$ and it also depends on $pi$(policy).

\

It follows from the properties of conditional expectation that

#align(center)[$
  underbrace(EE[ G_t | S_t = s ], v_pi (s))
  = sum_a underbrace(EE[ G_t | S_t = s, A_t = a ], q_pi (s,a)) pi(a|s)
$]

Hence,

#align(center)[
  #rect[*$ v_pi (s) = sum_a pi(a|s) q_pi (s,a) #h(2em) (1) $*]
]

(state value can be divided by different actions)
\
\
\

Recall that the state value is given by

#align(center)[$
  v_pi (s) = sum_a pi(a|s)
  underbrace(
    [ sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_pi (s') ],
    q_pi (s,a)
  )
$]

By comparing them, we have the action-value function as

#align(center)[
  *$ q_pi (s,a) = sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_pi (s') #h(1em) (2) $*]

\

Actually, the two expressions (1)&(2) shows two sides of the same coin:

① (1)shows how to obtain state values from action values.(this is simple because we just do average on the actions)\

② (2)shows how to obtain action values from state values.(this one is harder)

\
\

- An illustrative example

#figure(
  image("lec2_actionV_example.png", width: 40%),
)

With
#align(center)[
  $
    q_pi (s,a) = sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_pi (s')
  $]

we can get:
$ q_pi (s_1, a_2) = -1 + gamma v_pi (s_2) $

Notice: although the policy given asks to take action $a_2$ at $s_1$, we can get action values for _other actions_ !!!



#align(center)[$
  q_pi (s_1, a_1) & = -1 + gamma v_pi (s_1), \
  q_pi (s_1, a_3) & = 0 + gamma v_pi (s_3), \
  q_pi (s_1, a_4) & = -1 + gamma v_pi (s_1), \
  q_pi (s_1, a_5) & = 0 + gamma v_pi (s_1).
$]


\

~~~~① We can first calculate all the state values and then calculate the action values. \
~~~~② And we can also directly calculate the action values with or without models !
\
\
\
#rect[
  Q: Why can we compute $q_pi (s,a)$ for an action $a$ that $pi$(the policy) never takes at state $s$?
  \
  \
  A:
  $q_pi (s,a)$ conditions on $A_t = a$ only for the current step.
  After that, all future actions follow $pi$.

  - It does not require $pi(a|s) > 0$.
  - It answers: “What if we deviate *_now_*, then follow $pi$?”
  - This allows comparing actions to improve $pi$.

  Hence, evaluating actions not taken by $pi$ is both valid and essential.
]
