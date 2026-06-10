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













#pagebreak()
