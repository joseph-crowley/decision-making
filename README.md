# POMDP Solver

This Python program provides a comprehensive POMDP (Partially Observable Markov Decision Process) class that includes both the QMDP and PBVI solvers, along with additional methods for belief updates and action selection. The class also includes a method to simulate the POMDP environment to test the implemented solvers.

## Table of Contents

- [POMDP Class](#pomdp-class)
  - [Initialization](#initialization)
  - [Update Belief](#update-belief)
  - [QMDP Solver](#qmdp-solver)
  - [PBVI Solver](#pbvi-solver)
  - [Get Action](#get-action)
  - [Simulate](#simulate)
- [Test POMDP](#test-pomdp)

## POMDP Class

### Initialization

The `POMDP` class is initialized with the following parameters:

- `n_states`: The number of states in the POMDP.
- `n_actions`: The number of actions available to the agent.
- `n_observations`: The number of possible observations the agent can receive.
- `T`: The state transition function, a 3D array of size (n_states, n_actions, n_states).
- `Z`: The observation function, a 3D array of size (n_states, n_actions, n_observations).
- `R`: The reward function, a 2D array of size (n_states, n_actions).
- `gamma`: The discount factor, a scalar between 0 and 1.

### Update Belief

The `update_belief` method takes the current belief state, an action, and an observation as input and returns the updated belief state.

```python
def update_belief(self, belief, action, observation)
```

### QMDP Solver

The `qmdp` method computes the Q-values for each state-action pair using the QMDP algorithm. It takes an optional parameter `n_iterations` to specify the number of iterations to run the algorithm (default is 100).

```python
def qmdp(self, n_iterations=100)
```

### PBVI Solver

The `pbvi` method computes the value function using the Point-Based Value Iteration (PBVI) algorithm. It takes two optional parameters: `n_iterations` to specify the number of iterations to run the algorithm (default is 100), and `n_points` to specify the number of points in the value function (default is 10).

```python
def pbvi(self, n_iterations=100, n_points=10)
```

### Get Action

The `get_action` method finds the optimal action given a belief state. It takes the belief state and an optional parameter `solver` to specify the solver to use ('qmdp' or 'pbvi', default is 'qmdp'). If using the PBVI solver, additional optional parameters `n_iterations` and `n_points` can be specified.

```python
def get_action(self, belief, solver='qmdp', n_iterations=100, n_points=10)
```

### Simulate

The `simulate` method simulates the POMDP environment for a given number of steps and returns the total reward. It takes the initial belief state, the number of steps, and an optional parameter `solver` to specify the solver to use ('qmdp' or 'pbvi', default is 'qmdp'). If using the PBVI solver, additional optional parameters `n_iterations` and `n_points` can be specified.

```python
def simulate(self, initial_belief, n_steps, solver='qmdp', n_iterations=100, n_points=10)
```

## Test POMDP

The test POMDP has the following components:

- `n_states`: 2
- `n_actions`: 2
- `n_observations`: 2
- `T`: State transition function.
- `Z`: Observation function.
- `R`: Reward function.
- `gamma`: Discount factor (0.9).

The POMDP is instantiated, and the total reward is computed for both the QMDP and PBVI solvers using the `simulate` method.

```python
pomdp = POMDP(n_states, n_actions, n_observations, T, Z, R, gamma)

initial_belief = np.array([0.5, 0.5])
n_steps = 10

total_reward_qmdp = pomdp.simulate(initial_belief, n_steps, solver='qmdp')
total_reward_pbvi = pomdp.simulate(initial_belief, n_steps, solver='pbvi')

print("Total reward (QMDP):", total_reward_qmdp)
print("Total reward (PBVI):", total_reward_pbvi)
```