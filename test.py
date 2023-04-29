import unittest
import numpy as np
from POMDP import POMDP

class TestPOMDP(unittest.TestCase):
    
    def setUp(self):
        self.n_states = 2
        self.n_actions = 2
        self.n_observations = 2

        self.T = np.array([
            [[0.8, 0.2], [0.1, 0.9]],
            [[0.5, 0.5], [0.4, 0.6]]
        ])

        self.Z = np.array([
            [[0.8, 0.2], [0.1, 0.9]],
            [[0.6, 0.4], [0.2, 0.8]]
        ])

        self.R = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ])

        self.gamma = 0.95

        self.pomdp = POMDP(self.n_states, self.n_actions, self.n_observations, self.T, self.Z, self.R, self.gamma)
        self.initial_belief = np.array([0.5, 0.5])
        self.n_steps = 10

    def test_qmdp(self):
        total_reward = self.pomdp.simulate(self.initial_belief, self.n_steps, solver='qmdp')
        self.assertTrue(isinstance(total_reward, np.float64))
        
    def test_pbvi(self):
        total_reward = self.pomdp.simulate(self.initial_belief, self.n_steps, solver='pbvi')
        self.assertTrue(isinstance(total_reward, np.float64))
        
    def test_update_belief(self):
        belief = self.initial_belief
        action = 0
        observation = 0
        updated_belief = self.pomdp.update_belief(belief, action, observation)
        self.assertTrue(np.isclose(np.sum(updated_belief), 1.0))
        
    def test_invalid_solver(self):
        with self.assertRaises(ValueError):
            self.pomdp.simulate(self.initial_belief, self.n_steps, solver='invalid_solver')
        
if __name__ == '__main__':
    unittest.main()
