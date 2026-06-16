//全局配置 (放在最上面，只写一次)
#set page(
  paper: "us-letter",
  columns: 2,
  margin: (x: 1in, y: 1in), //缩减边距，让双栏更美观
  //设置页码的计数
  footer: context {
    let page_number = counter(page).at(here()).first()
    align(center, text(size: 9pt, font: "New Computer Modern")[
      #page_number
    ])
  },
)


//首页标题 (跨栏显示)
#place(top, scope: "parent", float: true)[
  #align(center)[
    #v(0.5in)
    #text(size: 25pt, weight: "bold")[Notes in RL(WestlakeU-zhao)]
    #v(1em)
    #text(size: 14pt)[Sean] \
    #text(size: 10pt)[#link("stone_breaker365@outlook.com")]
    #v(1em)
    #block(width: 90%, stroke: (y: 0.5pt), inset: 1em)[
      #set align(left)
      *Abstract* --- This note records Sean's notes of Westlake University course taught by Zhao Shiyu.
    ]
    #v(2em)
  ]
]













#pagebreak()





#place(top, scope: "parent", float: true)[
  #align(center + horizon)[  // horizon 让它垂直居中页顶区域，更美观
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec I]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Basic Concepts in RL
\
=== 1. Grid-World Example
\
*_State_*: The status of the agent with respect to the environment.\
~~~~In the grid world, the state is the location of the agent: $s_1, s_2 ...s_9$.\

#figure(
  image("lec1_state.png", width: 30%),
)
*_State space_*: $S = {s_i}_(i=1)^9$
\
\
\
*_Action_*: For each state, there are several possible actions: $a_1,...a_5$

#figure(
  image("lec1_action.png", width: 30%),
)
*_Action space of a state_*: $cal(A)(s_i) = {a_i}_(i=1)^5$
\
\
\
*_State trasition_*: When taking an action, the agent may move from one state to another. It actually defines the agent's interaction with the environment.
$ s_1 arrow^(a_2) s_2 $

~~~~We can use *_tabular representation_* to describe the state transition.(only deterministic cases)
#figure(
  image("lec1_tabular_representation.png", width: 70%),
)

*_Forbidden area_*: \
~~~~case 1: it's accessible but with penalty $✓$\
~~~~case 2: it's inaccessible (less general)
\
State transition probability:
$
  p(s_2|s_1, a_2) & = 1 \
  p(s_i|s_1, a_2) & = 0 #h(2em) forall i != 2
$
~~~~Here it is a deterministic case, but state transition can be stochastic.\
\
\

*_Policy_* tells the agent what actions to take at a state.\
~~~~We can use _conditional probability_ to describe a policy.(the probs of actions conditioned on each state)

#grid(
  columns: (1fr, 1fr, 1fr),
  figure(image("lec1_policy_arrows.png", width: 60%)),
  figure(image("lec1_deterministic_policy_prob.png", width: 45%)),
  figure(image("lec1_stochastic_policy_prob.png", width: 50%)),
)

~~~~There are stochastic policies as well.\
~~~~And we can use _tabular representation_ of a policy.\
#figure(
  image("lec1_tabular_repre_policy.png", width: 70%),
)

#rect[
  Q: How to implement a stochastic policy?\
  A: We can sample uniformly from $(0,1)$ and assign subintervals to actions based on their probabilities under the policy. The sampled value determines which action is taken.
]





#pagebreak()




*Reward*: a real number we get after taking an action.\
~~~~Usually: '$+$': encouragement~~~vs.~~ '$-$': punishment.\
~~~~In the grid-world example:
$ r_("bound") = -1, r_("forbid") = -1, r_("target") = +1, r_("other")=0 $

~~~~Reward can be interpreted as a _*human-machine interface*_, with whitch we can guide the agent to behave as what we want.\

~~~~Tabular representation of reward transition:\
#figure(
  image("lec1_tabular_repre_reward_trans.png", width: 70%),
)

~~~~Mathematical representation of reward by conditional probability.(it can also be stochastic)\
$ p(r=-1|s_1, a_1) = 1 "and" p(r!=-1|s_1, a_1) = 0 $
\
\
\

*Trajectory* is a state-action-reward chain.
$ s_1 arrow_(r = 0)^(a_2) s_2 arrow_(r = 0)^(a_3) s_5 arrow_(r = 0)^(a_3) s_8 arrow_(r = 1)^(a_2) s_9 $

#figure(
  image("lec1_trajectory_map.png", width: 35%),
)
~~~~The *_return_* of this trajectory is 0+0+0+1 = 1 (the sum of all the rewards collected along the trajectory)
\

~~~~*Return* can be used to evaluate whether a _policy_ is good or not.
\
\
\
\
*Discounted return*:\
~~~~Since a trajectory may be infinite,
$
  s_1 arrow_(r = 0)^(a_2) s_2 arrow_(r = 0)^(a_3) s_5 arrow_(r = 0)^(a_3) s_8 arrow_(r = 1)^(a_2) s_9 arrow_(r=1)^(a_5) s_9 arrow_(r=1)^(a_5) s_9......
$
the definition of the return is invalid when the return diverges ($infinity$)!
\

~~~~So we need to introduce a *_discount rate $gamma in [0, 1]$_*, then we get the discounted return :
$
  "discounted return" & = 0+gamma 0+ gamma^2 0+ gamma^3 1+ gamma^4 1+... \
                      & = gamma^3(1+gamma+gamma^2+...) = gamma^3 frac(1, 1-gamma)
$

~~~~If $gamma$ is close to 0, the value of the discounted return is dominated by the rewards obtained in the *_near_* future.\
~~~~If $gamma$ is close to 1, the value of the discounted return is dominated by the rewards obtained in the *_far_* future.\
\
\
\

*Episode*:\
~~~~When interacting with the environment following a policy, the agent may stop at some *_terminal states_*. The resulting trajectory  is called an episode.(or a trial)\

~~~~An episode is usually a *_finite_* trajectory. Tasks with episodes are called *_episodic tasks_*.
\
~~~~If the task has no terminal states, it's called _*continuing task*_.(the interaction with the environment will never end)
\
\

~~~~In fact, we can treat episodic and continuing tasks in a unified mathematical way by #underline[converting episodic tasks to continuing tasks].\
~~~~Op.① Treat the target state as a special absorbing state.(never leave and no more reward)\
~~~~Op.② Treat the target state as normal state with a policy.(can still leave and gain r = +1 when entering the target state) $✓$
\
\
\
\
\

=== 2. *MDP*(Markov decision process)
\
Key elements of MDP:

- *Sets:*
  - State: the set of states $S$
  - Action: the set of actions $cal(A)(s)$ is associated for state $s in S$
  - Reward: the set of rewards $cal(R)(s,a)$

- *Probability distribution:*
  - State transition probability: at state $s$, taking action $a$, the probability to transit to state $s'$ is $p(s' | s,a)$
  - Reward probability: at state $s$, taking action $a$, the probability to get reward $r$ is $p(r | s,a)$

- *Policy:* at state $s$, the probability to choose action $a$ is $pi(a|s)$

- *Markov property:* _memoryless_ property
  $ p(s_(t+1) | a_t, s_t, dots, a_0, s_0) = p(s_(t+1) | a_t, s_t) $
  $ p(r_(t+1) | a_t, s_t, dots, a_0, s_0) = p(r_(t+1) | a_t, s_t) $
\
Now we can understand MDP better:\
~~~~Markov: the memoryless property\
~~~~Decision: we have a policy\
~~~~Process: we have transitions(prob)\
\
\
\
The grid world could be abstracted as more general model, #underline[Markov process].\
#figure(
  image("lec1_grid_Markov_map.png", width: 65%),
)

Once a policy is given, the _Markov decision process_ becomes a _Markov process_.






#pagebreak()







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







#pagebreak()










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





#pagebreak()







#place(top, scope: "parent", float: true)[
  #align(center + horizon)[  // horizon 让它垂直居中页顶区域，更美观
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec IV]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Value Iteration and Policy Iteration
\
=== 1. Value iteration algorithm
\
- *Alogorithm description*:
$
  v = f(v) = max_π (r_π + gamma P_π v)
$
~~~~In the last lecture, we know that the contraction mapping theorem suggests an iterative algorithm to solve BOE:
$
  v_(k+1) = f(v_k) = max_π (r_π + gamma P_π v_k), k = 1,2,3, dots
$
where $v_0$ can be arbitrary.

~~~~This algorithm can eventually find the optimal state value $v^*$ and an optimal policy $pi^*$ and it is called *value iteration*!
\
\
- *Matrix-vector form*
~~~~Value iteration can be decomposed into two steps:
\
- - *Step 1: policy update*. This step is to solve
  $
    π_(k+1) = "arg max"_π (r_π + gamma P_π v_k)
  $
  where $v_k$ is given.

- - *Step 2: value update*.
  $
    v_(k+1) = r_(π_(k+1)) + gamma P_(π_(k+1)) v_k
  $

#rect[
  Q: Is $v_k$ a state value? \
  A: No, because it is not ensured that $v_k$ satisfies a Bellman equation. $v_k$ is just an arbitary valued vector.
]

Note that :\
~~~~Matrix-vector form is useful for theoretical analysis.\
~~~~Elementwise form is useful for implementation.






- *Elementwise form* :
\
- - *Step 1: policy update*

~~~~The elementwise form of
$
  π_(k+1) = "arg max"_π (r_π + gamma P_π v_k)
$
is
$
  π_(k+1)(s) = "arg max"_π sum_a π(a|s) underbrace(sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_k(s'), quad q_k(s,a)), s in cal(S)
$

~~~~The optimal policy solution is
$
  π_(k+1)(a|s) = cases(
    1 #h(1em) a = a_k^*(s),
    0 #h(1em) a != a_k^*(s)
  )
$
where $a_k^*(s) = "arg max"_a q_k(a,s)$. \
$π_(k+1)$ is called a *greedy policy*, since it simply selects the greatest $q$-value.

\
\
- - *Step 2: value update*

~~~~The elementwise form of
$
  v_(k+1) = r_(π_(k+1)) + gamma P_(π_(k+1)) v_k
$
is
$
  v_(k+1)(s) = sum_a π_(k+1)(a|s) underbrace(sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_k(s'), quad q_k(s,a)), s in cal(S)
$

~~~~Since $π_(k+1)$ is greedy, the above equation is simply
$
  v_(k+1)(s) = max_a q_k(s,a)
$




#pagebreak()




- *Procedure summary:*

#rect[
  *$ v_k(s) -> q_k(s,a) -> "greedy policy" π_(k+1)(a|s) \
  -> "new value" v_(k+1)(s) = max_a q_k(s,a) $*
]
\
\

- *Pseudocode: Value iteration algorithm*

- - Initialization: The probability model $p(r|s,a)$ and $p(s'|s,a)$ for all $(s,a)$ are known. Initial guess $v_0$.

- - Aim: Search the optimal state value and an optimal policy solving the Bellman optimality equation.
\
~~~~While $v_k$ has not converged in the sense that $||v_k - v_(k-1)||$ is greater than a predefined small threshold, for the $k$th iteration, do:

#rect[
  ~~~~ For every state $s in cal(S)$, do:\
  ~~~~~~~~For every action $a in cal(A)(s)$, do:\
  ~~~~~~~~~~~~*q-value*: $q_k(s,a) = sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_k(s')$\
  ~~~~~~~~*Maximum action value*: $a_k^*(s) = "arg max"_a q_k(s,a)$\
  ~~~~~~~~*Policy update*: $π_(k+1)(a|s) = 1$ if $a = a_k^*$, and $π_(k+1)(a|s) = 0$ otherwise\
  ~~~~~~~~*Value update*: $v_(k+1)(s) = max_a q_k(s,a)$
]


\
\
\
\
\
\
\
\
\
\
\
\
\

=== 2. Policy iteration algorithm
\
- *Algorithm description*:
~~~~Given a random initial policy $π_0$,
\
\
- - *Step 1: policy evaluation (PE)*
  ~~~~This step is to calculate the state value of $π_k$:
  $
    v_(π_k) = r_(π_k) + gamma P_(π_k) v_(π_k)
  $
  ~~~~$v_(π_k)$ is a state value function. (given a policy, we substitute it into the _Bellman equation_ to get the state value)
\
- - *Step 2: policy improvement (PI)*
  $
    π_(k+1) = "arg max"_π (r_π + gamma P_π v_(π_k))
  $
  The maximization is component-wise: \
  ~~~~For each state $s$, independently pick the action $a$ that maximizes the state–action value $r(s,a) + gamma sum_(s') P(s' | s,a) v_(pi_k)(s')$ . The resulting policy $pi_(k+1)$ is the concatenation of these per‑state optimal actions.
\

~~~~The algorithm leads to a sequence:
#rect[
  $
    π_0 ->^[P E] v_(π_0) arrow^[P I] π_1 ->^[P E] v_(π_1) ->^[P I] π_2 ->^[P E] v_(π_2) ->^[P I] dots
  $
]


\
#rect[
  Questions:\

  Q1: In the policy evaluation step, how to get the state value $v_(π_k)$ by solving the Bellman equation?\

  Q2: In the policy improvement step, why is the new policy $π_(k+1)$ better than $π_k$?\

  Q3: Why such an iterative algorithm can finally reach an optimal policy?\

  Q4: What is the relationship between this policy iteration algorithm and the previous value iteration algorithm?
]




#pagebreak()




#rect[
  Answers:\
  *A1*:\
  ~~~~In PE, how to get $v_(π_k)$ by solving the Bellman equation?
  $
    v_(π_k) = r_(π_k) + gamma P_(π_k) v_(π_k)
  $

  - Closed-form solution:
    $
      v_(π_k) = (I - gamma P_(π_k))^(-1) r_(π_k)
    $

  - Iterative solution:
    $
      v_(π_k)^((j+1)) = r_(π_k) + gamma P_(π_k) v_(π_k)^((j)), j = 0,1,2,dots
    $

  ~~~~Policy iteration is an iterative algorithm with another iterative algorithm embedded in the policy evaluation step!


  #v(0.5em)
  #line(length: 100%)
  #v(0.5em)
  *A2*:\
  ~~~~In PI, why is $π_(k+1)$ better than $π_k$?

  *Lemma (Policy Improvement)* \

  If $π_(k+1) = "arg max"_π (r_π + gamma P_π v_(π_k))$, then $v_(π_(k+1)) >= v_(π_k)$ for any $k$.
  \

  ~~~~It is provable.

  #v(0.5em)
  #line(length: 100%)
  #v(0.5em)

  *A3*: \
  ~~~~Why can such an iterative algorithm finally reach an optimal policy?

  ~~~~Since every iteration would improve the policy(from A2), we know
  $
    v_(π_0) <= v_(π_1) <= v_(π_2) <= dots <= v_(π_k) <= dots <= v^*.
  $
  ~~~~As a result, $v_(π_k)$ keeps increasing and will converge. Still need to prove it converges to $v^*$.
  \
  \
  *Theorem (Convergence of Policy Iteration)* \

  The state value sequence ${v_(π_k)}_(k=0)^infinity$ generated by the policy iteration algorithm converges to the optimal state value $v^*$. As a result, the policy sequence ${π_k}_(k=0)^infinity$ converges to an optimal policy.

]

#rect[
  *A4*:\

  ~~~~Policy iteration and Value iteration are two extreme cases of Truncated iteration.
]


\
\
\
\
\
\
- *Element-wise form*
\
- - *Step 1: PE*

① Matrix-vector form:
$
  v_(π_k)^((j+1)) = r_(π_k) + gamma P_(π_k) v_(π_k)^((j)), j = 0,1,2,dots
$

② Elementwise form:
$
  v_(π_k)^((j+1))(s) =
$
$ sum_a π_k(a|s) ( sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_(π_k)^((j))(s') ), s in cal(S) $

~~~~Stop when $j -> infinity$ or $j$ is sufficiently large or $||v_(π_k)^((j+1)) - v_(π_k)^((j))||$ is sufficiently small.

\
\

- - *Step 2: PI*

- Matrix-vector form:
  $
    π_(k+1) = "arg max"_π (r_π + gamma P_π v_(π_k))
  $

- Elementwise form:
  $
    π_(k+1)(s) =
  $
  $
    "arg max"_π sum_a π(a|s) underbrace(
      sum_r p(r|s,a) r
      + gamma sum_(s') p(s'|s,a) v_(π_k)(s'), quad q_(π_k)(s,a)
    ), s in cal(S)
  $

~~~~Here, $q_(π_k)(s,a)$ is the action value under policy $π_k$. Let
$
  a_k^*(s) = "arg max"_a q_(π_k)(s,a)
$
Then, the greedy policy is
$
  π_(k+1)(a|s) = cases(
    1 #h(1em) a = a_k^*(s),
    0 #h(1em) a != a_k^*(s)
  )
$
\
\
\

- *Pseudocode: Policy iteration algorithm*

- - Initialization: The probability model $p(r|s,a)$ and $p(s'|s,a)$ for all $(s,a)$ are known. Initial guess $π_0$.

- - Aim: Search for the optimal state value and an optimal policy.

~~~~While the policy has not converged, for the $k$th iteration, do:

#rect[
  ~~~~~~~~① *Policy evaluation*:\
  ~~~~~~~~Initialization: an arbitrary initial guess $v_(π_k)^((0))$\
  ~~~~~~~~While $v_(π_k)^((j))$ has not converged, for the $j$th iteration, do:\
  ~~~~~~~~~~~~For every state $s in cal(S)$, do:
  $
    v_(π_k)^((j+1))(s) =
  $
  $ sum_a π_k(a|s) [ sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_(π_k)^((j))(s') ] $

  ~~~~~~~~② *Policy improvement*:\
  ~~~~~~~~For every state $s in cal(S)$, do:\
  ~~~~~~~~~~~~For every action $a in cal(A)(s)$, do:
  $
    q_(π_k)(s,a) = sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_(π_k)(s')
  $
  ~~~~~~~~~~~~$a_k^*(s) = "arg max"_a q_(π_k)(s,a)$
  ~~~~~~~~~~~~$π_(k+1)(a|s) = 1$ if $a = a_k^*$, and $π_(k+1)(a|s) = 0$ otherwise
]

\
\
\
\
\
\
\

=== 3. Truncated policy iteration algorithm
\
- Compare value iteration and policy iteration

The two algorithms are very similar:
\

Policy iteration:
*$ π_0 ->^[P E] v_(π_0) ->^[P I] π_1 ->^[P E] v_(π_1) ->^[P I] π_2 ->^[P E] v_(π_2) ->^[P I] dots $*

Value iteration:
*$ u_0 ->^[P U] π'_1 ->^[V U] u_1 ->^[P U] π'_2 ->^[V U] u_2 ->^[P U] dots $*

PE = policy evaluation. \
PI = policy improvement. \
PU = policy update. \
VU = value update.
\

\
#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  // 表头
  table.header([], [*Policy iteration*], [*Value iteration*]),
  table.hline(),
  // 第一行
  [1) Policy:], [$π_0$], [N/A],
  table.hline(),
  // 第二行
  [2) Value:], [$v_(π_0) = r_(π_0) + gamma P_(π_0) v_(π_0)$], [$v_0 := v_(π_0)$],
  table.hline(),
  // 第三行
  [3) Policy:], [$π_1 = "arg max"_π (r_π + gamma P_π v_(π_0))$], [$π_1 = "arg max"_π (r_π + gamma P_π v_0)$],
  table.hline(),
  // 第四行
  [4) Value:], [$v_(π_1) = r_(π_1) + gamma P_(π_1) v_(π_1)$], [$v_1 = r_(π_1) + gamma P_(π_1) v_0$],
  table.hline(),
  // 第五行
  [5) Policy:], [$π_2 = "arg max"_π (r_π + gamma P_π v_(π_1))$], [$π'_2 = "arg max"_π (r_π + gamma P_π v_1)$],
  table.hline(),
  // 省略号
  [$\vdots$], [$\vdots$], [$\vdots$],
  table.hline(),
)

#v(1em)

- They start from the same initial condition, and the first three steps are the same. (Step 3 gives the same policy in both algorithms.)\

- The fourth step differs:
  - In *policy iteration*, solving $v_(π_1) = r_(π_1) + gamma P_(π_1) v_(π_1)$ requires an *iterative* algorithm (an embedded infinite number of iterations to give the solution of Bellman equation).
  - In *value iteration*, $v_1 = r_(π_1) + gamma P_(π_1) v_0$ is a *one-step* iteration.
- Consequently, $v_(π_1) >= v_1$ because $v_(π_1) >= v_(π_0)$.

\
#figure(
  image("lec4_truncated.png", width: 100%),
)
~~~~Therefore, the difference between value, truncated policy, policy iteration is mainly in the number of iterative steps of solving the Bellman equation.\
~~~~They just use different levels of solutions to update the policy(greedy), and then use the new policy to calculate the new $v$, and do all these steps iteratively.
\
\
\

- Thus we introduce *truncated policy iteration*.

#figure(
  image("lec4_truncated_policy_iteration_pseudocode.png", width: 100%),
)
\
\
~~~~Will the truncation undermine convergence?

*Proposition (Value Improvement)* \
~~~~Consider the iterative algorithm for solving the policy evaluation step:
$
  v_(π_k)^((j+1)) = r_(π_k) + gamma P_(π_k) v_(π_k)^((j)), j = 0,1,2,dots
$
~~~~If the initial guess is selected as $v_(π_k)^((0)) = v_(π_(k-1))$, it holds that
$
  v_(π_k)^((j+1)) >= v_(π_k)^((j))
$
for every $j = 0,1,2,dots$.
\
\

~~~~The convergence figure is like this:\
#figure(
  image("lec4_value_convergence.png", width: 50%),
)
~~~~The convergence proof of PI is based on that of VI. Since VI converges, we know PI converges.

\
\
\
\
\
\

*Summary:*
#figure(
  image("lec4_summary.png", width: 100%),
)





#pagebreak()










#place(top, scope: "parent", float: true)[
  #align(center + horizon)[  // horizon 让它垂直居中页顶区域，更美观
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec V]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Monte Carlo (MC) Learning
\
=== 1. Motivating example: MC estimation
\
~~~~Monte Carlo estimation refers to a broad class of techniques that rely on repeated random sampling to solve approximation problems.\
~~~~It does not require the model.\
\

*Law of Large Numbers*\
~~~~For a random variable $X$. Suppose ${x_j}_(j=1)^N$ are some iid samples. Let
$
  overline(x) = 1/N sum_(j=1)^N x_j
$
be the average of the samples. Then,
$
  E[overline(x)] = E[X], #h(2em)
  V a r[overline(x)] = 1/N V a r[X].
$
~~~~As a result, $overline(x)$ is an unbiased estimate of $E[X]$ and its variance decreases to zero as $N$ increases to infinity.

(The samples must be iid (independent and identically distributed))
\
\
\
~~~~Why mean estimation?\
~~~~Because state value and action
value are defined as expectations of random variables!

\
\
\
\
\
\
\

=== 2. MC Basic Algorithm
\
~~~~Policy iteration has two steps in each iteration:
$
  cases(
    "PE : " v_(π_k) = r_(π_k) + gamma P_(π_k) v_(π_k),
    "PI : " π_(k+1) = "arg max"_π (r_π + gamma P_π v_(π_k))
  )
$

~~~~The elementwise form of PI is:
$
  π_(k+1)(s) & = "arg max"_π sum_a π(a|s) [ sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_(π_k)(s') ] \
             & = "arg max"_π sum_a π(a|s) q_(π_k)(s,a), #h(2em) s in cal(S)
$

~~~~The key is $q_(π_k)(s,a)$! (Because the greedy policy will choose the greatest $q$)
\
\
\
- *Two expressions of action value*:

- - *Expression 1 requires the model*:
  $
    q_(π_k)(s,a) = sum_r p(r|s,a) r + gamma sum_(s') p(s'|s,a) v_(π_k)(s')
  $

- - *Expression 2 does not require the model*:
  $
    q_(π_k)(s,a) = EE[ G_t | S_t = s, A_t = a ]
  $

#rect[
  ~~~~Idea to achieve model-free RL: \
  ~~~~We can use expression 2 to calculate $q_(π_k)(s,a)$ based on _data_ (samples or experiences)!
]

\
\
\
- *The procedure of Monte Carlo estimation of action values*:

① Starting from $(s,a)$, following policy $π_k$, generate an episode.\
② The return of this episode is $g(s,a)$\
③ $g(s,a)$ is a sample of $G_t$ in
$
  q_(π_k)(s,a) = EE[ G_t | S_t = s, A_t = a ]
$
④ Suppose we have a set of episodes and hence ${g^((j))}(s,a)}$. Then,
$
  q_(π_k)(s,a) = EE[ G_t | S_t = s, A_t = a ] approx 1/N sum_(i=1)^N g^((i))}(s,a).
$

~~~~Fundamental idea: When model is unavailable, we can use _data_.

\
\
\
- *MC basic*

~~~~Description of the algorithm:

#rect[
  ~~~~Given an initial policy $π_0$, there are two steps at the $k$th iteration.

  - - Step 1: *policy evaluation*. This step is to #underline[_obtain $q_(π_k)(s,a)$ for all $(s,a)$_]. Specifically, for each action-state pair $(s,a)$, run an infinite number of (or sufficiently many) episodes. The #underline[average] of their returns is used to approximate $q_(π_k)(s,a)$.

  - - Step 2: *policy improvement*. This step is to solve $π_(k+1)(s) = "arg max"_π sum_a π(a|s) q_(π_k)(s,a)$ for all $s in cal(S)$. The greedy optimal policy is $π_(k+1)(a_k^* | s) = 1$ where $a_k^* = "arg max"_a q_(π_k)(s,a)$.
]
~~~~Exactly the same as the policy iteration algorithm, except:
- - Estimate $q_(π_k)(s,a)$ directly, instead of solving $v_(π_k)(s)$.


#figure(
  image("lec5_MCbasic_pseudocode.png", width: 100%),
)



#rect[
  MC Basic is a variant of the policy iteration algorithm.

  - The model-free algorithms are built up based on model-based ones. It is, therefore, necessary to understand model-based algorithms first before studying model-free algorithms.

  - MC Basic is useful to reveal the core idea of MC-based model-free RL, but not practical due to *_low efficiency_*.

  - Why does MC Basic estimate *_action values_* instead of *_state values_*? That is because state values cannot be used to improve policies directly. When models are not available, we should directly estimate action values.

  - Since policy iteration is convergent, the convergence of MC Basic is also guaranteed to be convergent given sufficient episodes.
]

\
\
\
\
\
\
\

=== 3. MC Exploring Starts
\
~~~~Consider a grid-world example, following a policy $π$, we can get an *episode* such as
$
  s_1 ->^(a_2) s_2 ->^(a_4) s_1 ->^(a_2) s_2 ->^(a_3) s_5 ->^(a_1) dots
$
\
- *Visit*: every time a *_state-action pair_* appears in the episode, it is called a visit of that state-action pair.
\
- Methods to use the data: *Initial-visit method*
  - Just calculate the return and approximate $q_π (s_1, a_2)$.
  - This is what the MC Basic algorithm does.
  - Disadvantage: #underline[Not fully utilize the data.]





#pagebreak()





~~~~The episode also visits other state-action pairs.

#figure(
  image("lec5_MC_exploring_stpairs.png", width: 100%),
)








~~~~Using this episode, we can estimate $q_π (s_1,a_2)$, $q_π (s_2,a_4)$, $q_π (s_2,a_3)$, $q_π (s_5,a_1)$, ...
\
\
#align(center)[
  #rect[
    *Improvements: Data-efficient methods*:\
    ① *first-visit method*\
    ② *every-visit method*
  ]]
\

- Another aspect in MC-based RL is *_when to update the policy_*. There are two methods.

- - ① The first method is, in the policy evaluation step, to collect _all the episodes_ starting from a state-action pair and then use the average return to approximate the action value. This is the one adopted by the MC Basic algorithm.

  - The problem of this method is that the agent has to wait until all episodes have been collected.
\
- - ② The second method uses the return of a _single episode_ to approximate the action value.
  - In this way, we can improve the policy _episode-by-episode_.

\
\
\
Will the second method cause problems?

~~~~One may say that the return of a single episode cannot accurately approximate the corresponding action value.\
~~~~In fact, we have done that in the truncated policy iteration algorithm introduced in the last chapter! (finite steps of approximation)

- *Generalized policy iteration (GPI):*
~~~~It's not a specific algorithm.\
~~~~It refers to the general idea or framework of #underline[switching between policy-_evaluation_ and policy-_improvement_ processes].\
~~~~Many model-based and model-free RL algorithms fall into this framework.

\
\
\
\
\

- *MC exploring starts*
\
~~~~If we combine improvements such as first/every-visit, episode‑by‑episode updates, and backward return accumulation, we get a new algorithm called MC Exploring Starts, yet it still relies on the *exploring‑starts assumption*.


#figure(
  image("lec5_MC_exploring_starts.png", width: 100%),
)

\
\
\
\
\
#rect[
  Q: What is *exploring starts*?\
  A: Exploring Starts (in Monte Carlo methods) is the assumption that *every episode has a non-zero probability of starting with any state-action pair $(s,a)$*. In theory, this allows us to begin a trajectory from any arbitrary state and action.
]




#pagebreak()




- Why do we need to consider exploring starts?

~~~~① In theory, only if every action value for every state is well explored, can we select the optimal actions correctly.\
~~~~On the contrary, if an action is not explored, this action may happen to be the optimal one and hence be missed.

~~~~② In practice, exploring starts is difficult to achieve. For many applications, especially those involving physical interactions with environments, it is difficult to collect episodes starting from every state-action pair.

~~~~Therefore, there is a gap between theory and practice.

~~~~Can we remove the requirement of exploring starts? We next show that we can do that by using _*soft policies*_.

\
\
\
\
\
\
\
\
\

=== 4. MC without exploring starts
\
~~~~Previously, we've introduced determistic and stochastic policies, here comes soft policies.
\

- *Soft policies*

- - A policy is called soft if #underline[the probability to take any action is positive.]

~~~~Why introduce soft policies?\
~~~~With a soft policy, a few episodes that are _sufficiently long_ can visit _every_ state-action pair for sufficiently many times.\
~~~~Then, we do not need to have a large number of episodes starting from every state-action pair. Hence, the requirement of exploring starts can thus be removed.




What soft policies will we use? \
Answer: *$epsilon$-greedy policies*

- What is an $epsilon$-greedy policy?
#align(center)[
  #rect[
    $
      pi(a|s) = cases(
        1 - frac(epsilon, |cal(A)(s)|) (|cal(A)(s)| - 1), "for the greedy action",
        frac(epsilon, |cal(A)(s)|), "for the other actions"
      )
    $
  ]]
~~~~where $epsilon in [0,1]$ and $|cal(A)(s)|$ is the number of actions for $s$.
\

- - The chance to choose the greedy action is always _greater_ than other actions, because
  $
    1 - epsilon / (|cal(A)(s)|) (|cal(A)(s)| - 1)) = 1 - epsilon + frac(epsilon, |cal(A)(s)|) >= epsilon / (|cal(A)(s)|).
  $
\
\
- Why use $epsilon$-greedy? \
~~~~Balance between *_exploitation_*(greedy) and *_exploration_*(less greedy).\

~~~~① When $epsilon = 0$, it becomes greedy! \
~~~~Less exploration but more exploitation!\
~~~~② When $epsilon = 1$, it becomes a uniform distribution. \
~~~~More exploration but less exploitation.

\
\

- How to embed $epsilon$-greedy into the MC-based RL algorithms?

- - *Originally*, the _policy improvement step_ in MC Basic and MC Exploring Starts is to solve
$
  π_(k+1)(s) = "arg max"_(π in Pi) sum_a π(a|s) q_(π_k)(s,a).
$
where #underline[$Pi$ denotes the set of all possible policies].
\

~~~~The optimal policy here is
$
  π_(k+1)(a|s) = cases(
    1 #h(1em)"if" a = a_k^*,
    0 #h(1em)"if" a != a_k^*,
  )
$
where $a_k^* = "arg max"_a q_(π_k)(s,a)$.




#pagebreak()





- - *Now*, the _policy improvement step_ is changed to solve
$
  π_(k+1)(s) = "arg max"_(π in Pi_epsilon) sum_a π(a|s) q_(π_k)(s,a),
$
where #underline[$Pi_epsilon$ denotes the set of all $epsilon$-greedy policies *_with a fixed value of $epsilon$_*.]

~~~~The optimal policy here is
$
  π_(k+1)(a|s) = cases(
    1 - (|cal(A)(s)| - 1)/(|cal(A)(s)|) epsilon #h(1em) "if" a = a_k^*,
    1/(|cal(A)(s)|) epsilon #h(4em) "if" a != a_k^*
  )
$

~~~~① MC $epsilon$-Greedy is the s*_ame_* as that of MC Exploring Starts except that the former uses $epsilon$-greedy policies.\

~~~~② It does not require exploring starts, but still requires to visit all state-action pairs in a different form.
\
\
#figure(
  image("lec5_MC_epsilon_greedy.png", width: 100%),
)

Notice: we use *_every-visit_* here because of the _sufficiently long_ episodes could contain many visits to a certain $(s, a)$.

\
\
\
\
\
\
\
\
\

- Exploring Ability
\
~~~~When ε = 1, the policy (uniform distribution) has the strongest
exploration ability.
\

~~~~When ε is small, the exploration ability of the policy is also small.

\
- Compared to greedy policies : \
~~~~① The advantage of ε-greedy policies is that they have stronger
exploration ability so that the exploring starts condition is not required.\

~~~~② The disadvantage is that ε-greedy polices are not optimal in general (we can only show that there always exist greedy policies that are optimal).\
~~~~The final policy given by the MC ε-Greedy algorithm is only optimal in the set Πε of all ε-greedy policies.\
~~~~ε cannot be too large.
\
~~~~In practical, we 'd set $epsilon$ to a _small_ value so that the final policy given is similar to that given by the optimal greedy policy.

\

- Consistency

~~~~The action of the biggest prob taken in the optimal $epsilon$-greedy policy is _perhaps_ in consistency with the optimal greedy policy.
\
~~~~As $epsilon$ grow bigger, that consistency descend!(that's why when we use $epsilon$-greedy we have to apply a small $epsilon$ !)
\







#pagebreak()













#place(top, scope: "parent", float: true)[
  #align(center + horizon)[  // horizon 让它垂直居中页顶区域，更美观
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec VI]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Stochastic Approximation and Stochastic Gradient Descent

\
=== 1. Motivating example: mean estimation
\
- Revisit the mean estimation problem:
- - Consider a random variable $X$. Our aim is to estimate $EE[X]$. Suppose that we collected a sequence of iid samples ${x_i}_{i=1}^N$.
- - The expectation of $X$ can be approximated by
  $
    EE[X] approx overline(x) := 1/N sum_(i=1)^N x_i.
  $

We already know from the last lecture:\
~~~~① This approximation is the basic idea of Monte Carlo estimation.\
~~~~② We know that $overline(x) -> EE[X]$ as $N -> infinity$.
\

Why do we care about mean estimation so much?\
~~~~Many values in RL such as state/action values are defined as means.

\
\
- New question: how to calculate the mean $overline(x)$?
$
  EE[X] approx overline(x) := 1/N sum_(i=1)^N x_i.
$

We have 2 ways.
\
\
~~~~① The first way, which is trivial, is to *collect all* the samples then calculate the average.\

~~~~The drawback of such way is that, if the samples are collected one by one over a period of time, we have to _wait_ until all the samples to be collected.
\
\

~~~~② The second way can avoid this drawback because it calculates the average in an incremental and iterative manner.
\
\
~~~~In particular, suppose
$
  w_(k+1) = 1/k sum_(i=1)^k x_i, quad k = 1,2,dots
$
and hence
$
  w_k = 1/(k-1) sum_(i=1)^(k-1) x_i, quad k = 2,3,dots
$
~~~~Then, $w_(k+1)$ can be expressed in terms of $w_k$ as
$
  w_(k+1) = 1/k sum_(i=1)^k x_i & = 1/k ( sum_(i=1)^(k-1) x_i + x_k ) \
                                & = 1/k ( (k-1) w_k + x_k ) \
                                & = w_k - 1/k (w_k - x_k).
$

~~~~Therefore, we obtain the following iterative algorithm:
#align(center)[
  #rect[
    *$ w_(k+1) = w_k - 1/k (w_k - x_k). $*
  ]]
\
~~~~This is an incremental form of calculation of $overline(x)$.


~~~~An advantage of this algorithm is that a mean estimate can be obtained _immediately_ once a sample is received. Then, the mean estimate can be used for other purposes immediately.\
~~~~The mean estimate is not accurate in the beginning due to insufficient samples (that is $w_k != EE[X]$). \
~~~~However, it is better than nothing. As more samples are obtained, the estimate can be improved gradually (that is $w_k -> EE[X]$ as $k -> infinity$).




#pagebreak()




- Furthermore, consider an algorithm with a more general expression:
#align(center)[
  #rect[
    $
      w_(k+1) = w_k - alpha_k (w_k - x_k)
    $
  ]]
where $1/k$ is replaced by $alpha_k > 0$.

- - Does this algorithm still converge to the mean $EE[X]$? We will show that the answer is yes if ${alpha_k}$ satisfy some mild conditions.

- - We will also show that this algorithm is a special *SA algorithm* and also a special *_stochastic gradient descent algorithm_*.

- - In the next lecture, we will see that the temporal-difference algorithms have similar (but more complex) expressions.


\
\
\
\
\
\

=== 2. Robins-Monro (RM) alogorithm
\

- *Stochastic approximation (SA)*

~~~~SA refers to a broad class of _stochastic iterative_ algorithms solving root finding or optimization problems.\

~~~~Compared to many other root-finding algorithms such as gradient-based methods, SA is powerful in the sense that *it does not require to know the _expression_ of the objective function nor its derivative*.
\
\
\

- *Robbins-Monro (RM) algorithm*
~~~~This is a pioneering work in the field of stochastic approximation.\
~~~~The famous stochastic gradient descent algorithm is a special form of the RM algorithm.
~~~~It can be used to analyze the mean estimation algorithms introduced in the beginning.



- Problem statement

~~~~Suppose we would like to find the root of the equation
$
  g(w) = 0,
$
where $w in RR$ is the variable to be solved and $g : RR -> RR$ is a function.\
\

~~~~① Many problems can be eventually converted to this root finding problem. For example, suppose $J(w)$ is an objective function to be minimized. Then, the _optimization problem_ can be converged to
$
  g(w) = nabla_w J(w) = 0.
$
~~~~② Note that an equation like $g(w) = c$ with $c$ as a constant can also be converted to the above equation by rewriting $g(w) - c = 0$ as a new function.
\
\
\

- #underline[How to calculate the root of $g(w) = 0$?]

- - If the expression of $g$ or its derivative is known, there are many numerical algorithms that can solve this problem.\

- - What if the expression of the function $g$ is *unknown*? For example, _*the function is represented by an artificial neuron network*_.

\
\
\

- The Robbins-Monro (RM) algorithm can solve this problem:
#align(center)[
  #rect[
    *$ w_(k+1) = w_k - a_k tilde(g)(w_k, eta_k), k = 1,2,3,dots $*]]

where\
~~~~① $w_k$ is the $k$th estimate of the root\
~~~~② *$tilde(g)(w_k, eta_k) = g(w_k) + eta_k$* is the $k$th _noisy_ observation ($eta_k$ is the noise)\
~~~~③ $a_k$ is a positive coefficient.



#pagebreak()



The function $g(w)$ is a *black box*! \
This algorithm relies on data:\
~~~~① _*Input sequence*_: ${w_k}$\
~~~~② _*Noisy output sequence*_: ${tilde(g)(w_k, eta_k)}$
\
\
~~~~Philosophy: without model, we need data!\
~~~~Here, the model refers to the expression of the function.


\
\
\

- *Convergence property*
\
- - Illustrative example

Solve:
$g(w) = tanh(w - 1)$ \
The true root of $g(w) = 0$ is $w^* = 1$.

Parameters: $w_1 = 3$, $a_k = 1/k$, $eta_k equiv 0$ (no noise for simplicity).

The RM algorithm in this case becomes
$
  w_(k+1) = w_k - 1/k g(w_k)
$
since $tilde(g)(w_k, eta_k) = g(w_k)$ when $eta_k = 0$.

#figure(
  image("lec6_RM_convergence_example.png", width: 60%),
)







result: $w_k$ converges to the true root $w^* = 1.0$.


Intuition: $w_(k+1)$ is closer to $w^*$ than $w_k$.

~~~① When $w_k > w^*$, we have $g(w_k) > 0$. Then $w_(k+1) = w_k - a_k g(w_k) < w_k$ and hence $w_(k+1)$ is closer to $w^*$ than $w_k$.\
~~~~② When $w_k < w^*$, we have $g(w_k) < 0$. Then $w_(k+1) = w_k - a_k g(w_k) > w_k$ and $w_(k+1)$ is closer to $w^*$ than $w_k$.

\


- - Convegence theorem
#figure(
  image("lec6_Robbins-Monro_theorem.png", width: 100%),
)

Explanation of the three conditions:

~~~~① $0 < c_1 <= nabla_w g(w) <= c_2$ for all $w$:\

~~~~$g$ is *monotonically increasing*, which ensures that the root of $g(w) = 0$ exists and is unique.\
~~~~The gradient is bounded from the above and must be positive(_convex_ original function).
\
\
~~~~② $sum_(k=1)^infinity a_k = infinity$ and $sum_(k=1)^infinity a_k^2 < infinity$\

~~~~The condition of $sum_(k=1)^infinity a_k^2 < infinity$ ensures that *$a_k$ converges to zero as $k arrow infinity$*.\

~~~~The condition of $sum_(k=1)^infinity a_k = infinity$ ensures that *$a_k$ do not converge to zero too fast*.
\
\
~~~~③ $EE[eta_k | cal(H)_k] = 0$ and $EE[eta_k^2 | cal(H)_k] < infinity$\

~~~~A special yet common case is that $\{eta_k\}$ is an *iid* stochastic sequence satisfying *$EE[eta_k] = 0$* and $EE[eta_k^2] < infinity$. The observation error $eta_k$ _is not required to be Gaussian_.


\
\
\

Examine the second condition more closely:
$
  sum_(k=1)^infinity a_k^2 < infinity #h(2em)
  sum_(k=1)^infinity a_k = infinity
$

~~~~*①* First, $sum_(k=1)^infinity a_k^2 < infinity$ indicates that $a_k -> 0$ as $k -> infinity$.\

~~~~Why is this condition important?
Since
$
  w_(k+1) - w_k = -a_k tilde(g)(w_k, eta_k),
$
~~~~If $a_k -> 0$, then $a_k tilde(g)(w_k, eta_k) -> 0$ and hence $w_(k+1) - w_k -> 0$.\
~~~~We need the fact that $w_(k+1) - w_k -> 0$ if $w_k$ converges eventually.\
~~~~If $w_k -> w^*$, $g(w_k) -> 0$ and $tilde(g)(w_k, eta_k)$ is dominated by $eta_k$.
\
\

~~~~*②* Second, $sum_(k=1)^infinity a_k = infinity$ indicates that $a_k$ should not converge to zero too fast.
\
\
~~~~Why is this condition important?

~~~~Summarizing $w_2 = w_1 - a_1 tilde(g)(w_1, eta_1)$, $w_3 = w_2 - a_2 tilde(g)(w_2, eta_2)$, ..., $w_(k+1) = w_k - a_k tilde(g)(w_k, eta_k)$ leads to
$
  w_infinity- w_1 = sum_(k=1)^infinity a_k tilde(g)(w_k, eta_k).
$

~~~~Suppose $w_infinity = w^*$. If $sum_(k=1)^infinity a_k < infinity$, then $sum_(k=1)^infinity a_k tilde(g)(w_k, eta_k)$ may be _bounded_. Then, if the initial guess $w_1$ is chosen arbitrarily far away from $w^*$, the above equality would be invalid.
\
~~~~Thus, it enables us to _choose the intial guess $w_1$ arbitarily_.

\
\

Recall our typical choice:
$ a_k = 1/k $
It satifies that $sum_(k=1)^infinity 1/k arrow infinity$ and $sum_(k=1)^infinity 1/k^2 < infinity$.

\
\

~~~~Note that these three are _sufficient but not necessary conditions_.\
~~~~If the three conditions are not satisfied, the algorithm may not work.\

~~~~For example, $g(w) = w^3 −5$ does not satisfy the first condition on gradient boundedness. If the initial guess is good, the algorithm can converge (locally). Otherwise, it will diverge.\

~~~~We will see that $a_k$ is often selected as a *sufficiently small constant* in
many RL algorithms. Although the second condition is not satisfied in this case, the algorithm can still work effectively.

\
\
\
\

- *RM's Application to mean estimation*
\
- - Recall that
$
  w_(k+1) = w_k + alpha_k (x_k - w_k)
$
is the mean estimation algorithm.

We know that\
~~~~If $alpha_k = 1/k$, then $w_(k+1) = 1/k sum_(i=1)^k x_i$.\
~~~~If $alpha_k$ is not $1/k$, the convergence was not analyzed.\

Next, we show that this algorithm is *_a special case of the RM algorithm_*. Then, its convergence naturally follows.

\
\

- -
~~~~1) Consider a function:
$
  g(w) := w - EE[X].
$
~~~~Our aim is to solve $g(w) = 0$. If we can do that, then we can obtain $EE[X]$.
\
\

~~~~2) The observation(noisy samples) we can get is (because we can only obtain samples of $X$)
$
  tilde(g)(w,x) := w - x,
$

Note that
$
  tilde(g)(w, eta) = w - x & = w - x + EE[X] - EE[X] \
                           & = (w - EE[X]) + (EE[X] - x) \
                           & := g(w) + eta.
$



~~~~3) The RM algorithm for solving $g(x) = 0$ is
$
  w_(k+1) = w_k - alpha_k tilde(g)(w_k, eta_k) = w_k - alpha_k (w_k - x_k)
$
which is exactly the mean estimation algorithm. The convergence naturally follows.

\
\
#rect[
  ~~~~To prove that the mean estimation with a step size other than $1/k$ is a special case of the RM algorithm, the reasoning is as follows:

  ~~~~The RM algorithm iteratively finds the root of $g(w)=0$.

  ~~~~We reformulate the mean estimation problem as finding the root of $g(w)=w−E[X]=0$.

  ~~~~Given observed samples $x$, we define a computable function $tilde(g)(w,x)=w−x$ (where $w$ is the current estimate and $x$ is the new sample).

  ~~~~Its expectation is $E[tilde(g)(w,x)]=g(w)$, so it meets the prerequisite for applying RM.

  ~~~~Therefore, applying the RM algorithm yields the mean estimation algorithm.
]

\
\
\
#rect[
  Conclusion:\
  Mean estimation algorithm = RM algorithm applied to this specific problem: $ g(w)=w−E[X]=0 $
]

\
\
\
\
\
\
\
\
- - Dvoretzky's theorem
#figure(
  image("lec6_Dvoretzky_theorem.png", width: 100%),
)

~~~~A more general result than the RM theorem. It can be used to prove the RM theorem.\
~~~~It can also directly analyze the mean estimation problem.\
~~~~An extension of it can be used to analyze Q-learning and TD learning algorithms.


\
\
\
\
\
\
\

=== 3. Stochastic gradient descent
\

~~~~Suppose we aim to solve the following optimization problem:
$
  min_w J(w) = EE[ f(w, X) ]
$

~~~~$w$ is the parameter to be optimized.\
~~~~$X$ is a random variable. The expectation is with respect to $X$. (X has a fixed but unknown probability distribution)\
~~~~$w$ and $X$ can be either scalars or vectors. The function $f(·)$ is a scalar.
\
\
\
- *Algorithm*

- - *Method 1: gradient descent (GD)*

$
  w_(k+1) = w_k - alpha_k nabla_w EE[ f(w_k, X) ] = w_k - alpha_k EE[ nabla_w f(w_k, X) ]
$

~~~~Drawback: the *_expected value_* is difficult to obtain.

- - *Method 2: batch gradient descent (BGD)*

$
  EE[ nabla_w f(w_k, X) ] approx 1/n sum_(i=1)^n nabla_w f(w_k, x_i)
$

$
  w_(k+1) = w_k - alpha_k 1/n sum_(i=1)^n nabla_w f(w_k, x_i)
$
~~~~Drawback: it requires _many samples in each iteration for each $w_k$_. (similar to Monte-Carlo method)
\

- - *Method 3: stochastic gradient descent (SGD)*

#align(center)[
  #rect[
    *$ w_(k+1) = w_k - alpha_k nabla_w f(w_k, x_k), $*
  ]]
~~~~Compared to GD: Replace the *true gradient* $EE[ nabla_w f(w_k, X) ]$ by the *stochastic gradient* $nabla_w f(w_k, x_k)$.\

~~~~Compared to BGD: let *$n = 1$*.

\
\
\
\
\
\

- *SGD - Example and application*
\
We next consider an example:
$
  min_w J(w) = EE[ f(w, X) ] = EE[ 1/2 ||w - X||^2 ]
$
where
$
  f(w, X) = (||w - X||^2) / 2, quad nabla_w f(w, X) = w - X.
$

\

Exercise 1: Show that the optimal solution is $w^* = EE[X]$.\

Exercise 2: Write out the GD algorithm for solving this problem.\

Exercise 3: Write out the SGD algorithm for solving this problem.


Answer 1:\
~~~~When achieving the optimal solution, we have $nabla_w J(w) = 0$, that is $EE[nabla_w f(w, x)] = 0$. Thus $EE[w-X] = 0$, $w^* = EE[X]$.
\
\
Answer 2: \
~~~~The GD algorithm for solving the above problem is
$
  w_(k+1) & = w_k - alpha_k nabla_w J(w_k) \
          & = w_k - alpha_k EE[ nabla_w f(w_k, X) ] \
          & = w_k - alpha_k EE[ w_k - X ].
$

Answer 3:\
~~~~The SGD algorithm for solving the above problem is
$
  w_(k+1) = w_k - alpha_k nabla_w f(w_k, x_k) = w_k - alpha_k (w_k - x_k).
$
\
~~~~Note that the SGD algorithm is the same as the mean estimation algorithm we presented before. *That mean estimation algorithm is a special SGD algorithm.*\
~~~~(the problem descriptions are different: before it asked for the mean of $X$, now it is described as an optimization problem).

\
\
\
\
\

- *Convergence analysis*
\
- - *From GD to SGD:*


$
  w_(k+1) = w_k - alpha_k EE[ nabla_w f(w_k, X) ]
$
#align(center)[$↓$]
$
  w_(k+1) = w_k - alpha_k nabla_w f(w_k, x_k)
$
$nabla_w f(w_k, x_k)$ can be viewed as a _noisy measurement_ of $EE[ nabla_w f(w, X) ]$:
$
  nabla_w f(w_k, x_k) = EE[ nabla_w f(w, X) ] + underbrace(nabla_w f(w_k, x_k) - EE[ nabla_w f(w, X) ], eta)
$

Since
$
  nabla_w f(w_k, x_k) != EE[ nabla_w f(w, X) ],
$
*whether $w_k -> w^*$ as $k -> infinity$ by SGD?*

\
\

- - *Proof thoughtline*:

~~~~We'll show that #underline[*SGD is a special RM algorithm*]. Then, the convergence naturally follows.
\
\
~~~~*The aim of SGD* is to minimize
$
  J(w) = EE[ f(w, X) ]
$
~~~~This problem can be converted to a *_root-finding_* problem:
$
  nabla_w J(w) = EE[ nabla_w f(w, X) ] = 0
$
Let
$
  g(w) = nabla_w J(w) = EE[ nabla_w f(w, X) ].
$
~~~~Then, the aim of SGD is to #underline[find the root of $g(w) = 0$.]
\
\
~~~~What we can measure is
$
  tilde(g)(w, eta) & = nabla_w f(w, x) \
                   & = underbrace(EE[ nabla_w f(w, X) ], g(w)) + underbrace(nabla_w f(w, x) - EE[ nabla_w f(w, X) ], eta)
$

~~~~Then, the RM algorithm for solving $g(w) = 0$ is
$
  w_(k+1) = w_k - a_k tilde(g)(w_k, eta_k) = w_k - a_k nabla_w f(w_k, x_k).
$
\
~~~~It is exactly the SGD algorithm.\
~~~~Therefore, SGD is a special RM algorithm.

\
\
\
\
\
\
- - *SGD convergence theorem*
#figure(
  image("lec6_SGD_convergence_theorem.png", width: 100%),
)
\

\
\
\
\
\

- *SGD convergence pattern*
\


*Question*: \
~~~~Since the stochastic gradient is random and hence the approximation is inaccurate, *_whether the convergence of SGD is slow or random?_*
\
\
- - To answer this question, we consider the *relative error* between the _stochastic_ and _batch gradients_:
$
  delta_k := (|nabla_w f(w_k, x_k) - EE[ nabla_w f(w_k, X) ]|) / (|EE[ nabla_w f(w_k, X) ]|).
$

~~~~Since $EE[ nabla_w f(w^*, X) ] = 0$ ($w^*$ is the optimal solution), we further have
$
  delta_k & = (|nabla_w f(w_k, x_k) - EE[ nabla_w f(w_k, X) ]|) / (|EE[ nabla_w f(w_k, X) ] - EE[ nabla_w f(w^*, X) ]|) \
          & = (|nabla_w f(w_k, x_k) - EE[ nabla_w f(w_k, X) ]|) / (|EE[ nabla^2_w f(tilde(w)_k, X) ] (w_k - w^*)|),
$
where the last equality is due to the _Lagrange's mean value theorem_ and $tilde(w)_k in [w_k, w^*]$
\
\
\

~~~~Suppose $f$ is strictly convex such that
$
  nabla^2_w f >= c > 0
$
for all $w, X$, where $c$ is a positive bound.

~~~~Then, the denominator of $delta_k$ becomes
$
  |EE[ nabla^2_w f(tilde(w)_k, X) ] (w_k - w^*)| & = |EE[ nabla^2_w f(tilde(w)_k, X) ]| |w_k - w^*| \
                                                 & >= c |w_k - w^*|.
$

~~~~Substituting the above inequality to $delta_k$ gives
$
  delta_k <= (|nabla_w f(w_k, x_k) - EE[ nabla_w f(w_k, X) ]|) / (c |w_k - w^*|).
$\

Note that
#align(center)[
  #rect[
    $
      delta_k <= (| overbrace(nabla_w f(w_k, x_k), "stochastic gradient") - overbrace(EE[ nabla_w f(w_k, X) ], "true gradient") |) / ( underbrace(c|w_k - w^*|, "distance to the optimal solution")).
    $
  ]]

~~~~The above equation suggests an interesting convergence pattern of SGD.

~~~~① The relative error $delta_k$ is inversely proportional to $|w_k - w^*|$.\
~~~~② *When $|w_k - w^*|$ is large, $delta_k$ is small and SGD behaves like GD.*\
~~~~③ *When $w_k$ is close to $w^*$, the relative error may be large and the convergence exhibits more randomness in the neighborhood of $w^*$*.

\
\
\
- - *Illustrative example*
\
~~~~*Setup*: $X in RR^2$ represents a random position in the plane. Its distribution is _uniform_ in the square area centered at the origin with the side length as 20. The true mean is $EE[X] = 0$. The mean estimation is
based on 100 iid samples ${x_i}^100_(i=1)$.


#figure(
  image("lec6_SGD_convergence_pattern_example.png", width: 80%),
)

~~~~Although the initial guess of the mean is far away from the true value, the SGD estimate can approach the neighborhood of the true value fast.\
~~~~When the estimate is close to the true value, it exhibits certain randomness but still approaches the true value gradually.\
\
\
\
\

- - *A deterministic formulation*\
\
~~~~The formulation of SGD we introduced above involves random variables($X$) and expectation. \
~~~~One may often encounter a deterministic formulation of SGD without involving any random variables.
\
\
\

~~~~Consider the optimization problem:
$
  min_w J(w) = 1/n sum_(i=1)^n f(w, x_i),
$
where\
~~~~① $f(w, x_i)$ is a parameterized function.\
~~~~② $w$ is the parameter to be optimized.\
~~~~③ a set of real numbers ${x_i}_(i=1)^n$, where $x_i$ #underline[does not have to be a sample of any random variable.]

\

~~~~The GD for solving this problem is
$
  w_(k+1) & = w_k - alpha_k nabla_w J(w_k) \
          & = w_k - alpha_k 1/n sum_(i=1)^n nabla_w f(w_k, x_i).
$

~~~~Suppose the set is large and we can only fetch _a single number every time_. In this case, we can use the following iterative algorithm (we only fetch *$x_k$* out of the set and use it) :
$
  w_(k+1) = w_k - alpha_k nabla_w f(w_k, x_k).
$
\



#pagebreak()




Questions:\
~~~~① Is this algorithm SGD? #underline[It does not involve any random variables or expected values.]\
~~~~② How should we use the finite set of numbers ${x_i}_(i=1)^n$? Should we sort these numbers in a certain order and then use them one by one? Or should we randomly sample a number from the set?

\

~~~~A quick answer to the above questions is that we can *_introduce a random variable manually_* and convert the deterministic formulation to the stochastic formulation of SGD.

~~~~In particular, suppose $X$ is _a random variable defined on the set ${x_i}_(i=1)^n$_. \
~~~~Suppose its probability distribution is _*uniform*_ such that
$
  p(X = x_i) = 1/n.
$
~~~~Then, the deterministic optimization problem becomes a stochastic one:
$
  min_w J(w) = 1/n sum_(i=1)^n f(w, x_i) = EE[ f(w, X) ].
$

~~~~The last equality in the above equation is _strict_ instead of approximate. Therefore, the algorithm is SGD.\
~~~~#underline[The estimate converges if *_$x_k$ is uniformly and independently sampled from ${x_i}_{i=1}^n$_*]. $x_k$ may repeatedly take the same number in ${x_i}_{i=1}^n$ since it is sampled randomly.\

#underline[(Note that the uniform sampling ensures $EE(eta) = 0$, satisfying the convergence condition of RM)]

\
\
\
\
\
\
\
\
- *BGD, MBGD(minibatch), SGD*
\
- - Suppose we would like to minimize $J(w) = EE[ f(w, X) ]$, given a set of random samples ${x_i}_(i=1)^n$ of $X$. The BGD, SGD, MBGD algorithms solving this problem are, respectively,

$
  w_(k+1) = w_k - alpha_k 1/n sum_(i=1)^n nabla_w f(w_k, x_i), (B G D)
$
$
  w_(k+1) = w_k - alpha_k 1/m sum_(j in I_k) nabla_w f(w_k, x_j), (M B G D)
$
$
  w_(k+1) = w_k - alpha_k nabla_w f(w_k, x_k). (S G D)
$
\
~~~~① In the BGD algorithm, all the samples are used in every iteration. When $n$ is large, $(1/n) sum_(i=1)^n nabla_w f(w_k, x_i)$ is close to the true gradient $EE[ nabla_w f(w_k, X) ]$.

~~~~② In the MBGD algorithm, *$I_k$ is a subset of ${1, dots, n}$ with the size as $|I_k| = m$*. The set $I_k$ is obtained by $m$ times iid samplings.

~~~~③ In the SGD algorithm, $x_k$ is randomly sampled from ${x_i}_{i=1}^n$ at time $k$.

\
\
- - *Compare MBGD with BGD and SGD*

~~~~① Compared to SGD, MBGD has _less randomness_ because it uses more samples instead of just one as in SGD.

~~~~② Compared to BGD, MBGD does not require to use all the samples in every iteration, making it _more flexible and efficient_.
\
\
(i) If $m = 1$, MBGD becomes SGD.\
(ii) If $m = n$, MBGD *does not become BGD strictly speaking* because *MBGD uses randomly fetched $n$ samples* whereas *BGD uses all $n$ numbers*. In particular, MBGD may use a value in ${x_i}_(i=1)^n$ multiple times whereas BGD uses each number once.




- - *Illustrative examples*

~~~~Given some numbers ${x_i}_(i=1)^n$, our _*aim is to*_\
_*calculate the mean*_ $overline(x) = sum_(i=1)^n (x_i \/ n)$. This problem can be equivalently stated as the following optimization problem (to minimize the variance!):

$
  min_w J(w) = 1/(2n) sum_(i=1)^n || w - x_i ||^2
$

~~~~The three algorithms for solving this problem are, respectively,

$
  w_(k+1) & = w_k - alpha_k 1/n sum_(i=1)^n (w_k - x_i) \
          & = w_k - alpha_k (w_k - overline(x)), (B G D)
$
$
  w_(k+1) & = w_k - alpha_k 1/m sum_(j in I_k) (w_k - x_j) \
          & = w_k - alpha_k ( w_k - overline(x)_k^((m)) ), (M B G D)
$
$
  w_(k+1) = w_k - alpha_k (w_k - x_k), (S G D)
$
\
where $overline(x)_k^((m)) = sum_(j in I_k) x_j / m$.

\
~~~~Furthermore, if $alpha_k = 1/k$, the above equation can be solved as

$
  w_(k+1) = 1/k sum_(j=1)^k overline(x) = overline(x), (B G D)
$
$
  w_(k+1) = 1/k sum_(j=1)^k overline(x)_j^((m)), (M B G D)
$
$
  w_(k+1) = 1/k sum_(j=1)^k x_j. (S G D)
$
\
~~~~① The estimate of BGD at each step is exactly the optimal solution $w^* = overline(x)$.\
~~~~② The estimate of MBGD approaches the mean faster than SGD because $overline(x)_k^((m))$ is already an average.



#figure(
  image("lec6_minibatch_GD_example.png", width: 100%),
)


\
\
\
\
\
#rect[
  - *Summary: Connections between mean estimation, RM, and SGD*

  - - Mean estimation: compute $EE[X]$ *using ${x_k}$*
  $
    w_(k+1) = w_k - 1/k (w_k - x_k).
  $

  - - RM algorithm: solve $g(w) = 0$ *using ${tilde(g)(w_k, eta_k)}$*(noisy observation)
  $
    w_(k+1) = w_k - a_k tilde(g)(w_k, eta_k).
  $

  - - SGD algorithm: minimize $J(w) = EE[ f(w, X) ]$ *using ${nabla_w f(w_k, x_k)}$*
  $
    w_(k+1) = w_k - alpha_k nabla_w f(w_k, x_k).
  $

  These results are useful:\
  ~~~~① We will see in the next chapter that the _temporal-difference_ learning algorithms can be viewed as stochastic approximation algorithms and hence have similar expressions.\
  ~~~~② They are important optimization techniques that can be applied to many other fields.
]








#pagebreak()










#place(top, scope: "parent", float: true)[
  #align(center + horizon)[  // horizon 让它垂直居中页顶区域，更美观
    #text(font: "Georgia", weight: "bold", size: 24pt)[§ Lec VII]  //
    #v(0em)
    #line(length: 100%, stroke: 1pt)  // 可选：加一条装饰线
  ]
]

== Temporal-Difference Learning




























