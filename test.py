import unittest
import numpy as np

class TestPOMDP(unittest.TestCase):

    def setUp(self):
        n_states = 3
        n_actions = 2
        n_observations = 2
        T = np.array([[[0.7, 0.3], [0.3, 0.7]], 
                      [[0.99, 0.01], [0.01, 0.99]], 
                      [[0.5, 0.5], [0.5, 0.5]]])
        Z = np.array([[[0.9, 0.1], [0.1, 0.9]], 
                      [[0.8, 0.2], [0.2, 0.8]], 
                      [[0.5, 0.5], [0.5, 0.5]]])
        R = np.array([[0.0, -0.5], 
                      [1.0, 0.0], 
                      [0.0, 0.0]])
        gamma = 0.95

        self.pomdp = POMDP(n_states, n_actions, n_observations, T, Z, R, gamma)

    def test_update_belief(self):
        belief = np.array([0.5, 0.4, 0.1])
        action = 0
        observation = 1
        expected_updated_belief = np.array([0.16981132, 0.64150943, 0.18867925])

        updated_belief = self.pomdp.update_belief(belief, action, observation)
        np.testing.assert_almost_equal(updated_belief, expected_updated_belief, decimal=8)

    def test_qmdp(self):
        n_iterations = 100
        expected_Q = np.array([[0.82745098, -0.17254902],
                               [1.82745098,  0.82745098],
                               [0.82745098,  0.82745098]])

        Q = self.pomdp.qmdp(n_iterations)
        np.testing.assert_almost_equal(Q, expected_Q, decimal=8)

    def test_pbvi(self):
        n_iterations = 100
        n_points = 10
        expected_value_function_shape = (n_points, self.pomdp.n_states)

        value_function = self.pomdp.pbvi(n_iterations, n_points)
        self.assertEqual(value_function.shape, expected_value_function_shape)

    def test_get_action(self):
        belief = np.array([0.5, 0.4, 0.1])
        solver = 'qmdp'
        n_iterations = 100
        n_points = 10
        expected_action = 0

        action = self.pomdp.get_action(belief, solver, n_iterations, n_points)
        self.assertEqual(action, expected_action)

    def test_simulate(self):
        initial_belief = np.array([0.5, 0.4, 0.1])
        n_steps = 10
        solver = 'qmdp'
        n_iterations = 100
        n_points = 10

        total_reward = self.pomdp.simulate(initial_belief, n_steps, solver, n_iterations, n_points)
        self.assertIsInstance(total_reward, float)

if __name__ == "__main__":
    unittest.main()