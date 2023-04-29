import numpy as np

class POMDP:
    def __init__(self, n_states, n_actions, n_observations, T, Z, R, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_observations = n_observations
        self.T = T
        self.Z = Z
        self.R = R
        self.gamma = gamma

    def update_belief(self, belief, action, observation):
        updated_belief = np.zeros(self.n_states)
        for s in range(self.n_states):
            updated_belief[s] = self.Z[action, observation, s] * np.sum(self.T[:, action, s] * belief)
        return updated_belief / np.sum(updated_belief)

    def qmdp(self, n_iterations=100):
        Q = np.zeros((self.n_states, self.n_actions))
        for _ in range(n_iterations):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    Q[s, a] = self.R[s, a] + self.gamma * np.sum(self.T[:, s, a] * np.max(Q, axis=1))
        return Q

    def pbvi(self, n_iterations=100, n_points=10):
        value_function = np.random.rand(n_points, self.n_states)

        for _ in range(n_iterations):
            alpha_vectors = np.zeros((self.n_actions, n_points, self.n_states))
            for a in range(self.n_actions):
                for i in range(n_points):
                    for s in range(self.n_states):
                        alpha_vectors[a, i, s] = self.R[s, a] + self.gamma * np.sum(self.T[s, a, :] * value_function[i, :])

            new_value_function = np.zeros_like(value_function)
            for i, point in enumerate(value_function):
                max_action = np.argmax(np.sum(alpha_vectors * point, axis=2), axis=0)
                new_value_function[i] = alpha_vectors[max_action[i], i, :]

            value_function = new_value_function

        return value_function

    def get_action(self, belief, solver='qmdp', n_iterations=100, n_points=10):
        if solver == 'qmdp':
            Q = self.qmdp(n_iterations)
            return np.argmax(np.dot(belief, Q))
        elif solver == 'pbvi':
            value_function = self.pbvi(n_iterations, n_points)
            action_values = np.dot(value_function, belief)
            return np.argmax(np.max(action_values, axis=0))  # select the action that gives the highest expected reward
        else:
            raise ValueError("Invalid solver. Choose either 'qmdp' or 'pbvi'.")

    def simulate(self, initial_belief, n_steps, solver='qmdp', n_iterations=100, n_points=10):
        belief = initial_belief
        total_reward = 0

        for _ in range(n_steps):
            action = self.get_action(belief, solver, n_iterations, n_points)
            reward = np.dot(belief, self.R[:, action])
            total_reward += reward

            true_state = np.random.choice(self.n_states, p=belief)
            next_state = np.random.choice(self.n_states, p=self.T[true_state, action])
            observation = np.random.choice(self.n_observations, p=self.Z[action, next_state])

            belief = self.update_belief(belief, action, observation)

        return total_reward