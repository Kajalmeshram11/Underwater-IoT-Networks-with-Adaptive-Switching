"""
=============================================================================
Q-LEARNING ADAPTIVE SWITCHING AGENT
=============================================================================
Implements a tabular Q-Learning agent that decides:
  ACTION 0 → Use OPTICAL communication
  ACTION 1 → Use ACOUSTIC communication
  ACTION 2 → Use HYBRID (both channels with data splitting)

State space is discretized from continuous channel observations.
Reward function balances: throughput, energy efficiency, latency, PDR.
=============================================================================
"""

import numpy as np
import random
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────────────────────────────────────────
ACTION_OPTICAL  = 0
ACTION_ACOUSTIC = 1
ACTION_HYBRID   = 2
ACTION_NAMES    = {0: 'Optical', 1: 'Acoustic', 2: 'Hybrid'}

N_ACTIONS = 3

def _zero_q_row():
    return np.zeros(N_ACTIONS)


# ─────────────────────────────────────────────────────────────────────────────
# STATE DISCRETIZER
# ─────────────────────────────────────────────────────────────────────────────
class StateDiscretizer:
    """
    Converts continuous channel state → discrete (hashable) state tuple.
    Discretization bins are chosen to reflect meaningful boundaries
    for underwater communication decisions.
    """

    # Distance bins (meters): short / medium / long / very-long
    DIST_BINS    = [0, 50, 100, 500, 1000, 5000]
    # Energy bins (Joules): critical / low / medium / high / full
    ENERGY_BINS  = [0, 5, 20, 50, 80, 100]
    # Data size bins (KB)
    DATA_BINS    = [0, 10, 50, 200, 500, 1000]
    # Turbidity bins
    TURB_BINS    = [0, 0.5, 1.0, 2.0, 5.0]

    def discretize(self, channel_state):
        """Returns a hashable discrete state tuple."""
        d_bin   = np.digitize(channel_state.distance,  self.DIST_BINS)
        e_bin   = np.digitize(channel_state.energy,    self.ENERGY_BINS)
        data_bin= np.digitize(channel_state.data_size, self.DATA_BINS)
        t_bin   = np.digitize(channel_state.turbidity, self.TURB_BINS)
        opt_ok  = int(channel_state.optical_feasible())
        aco_ok  = int(channel_state.acoustic_feasible())
        return (d_bin, e_bin, data_bin, t_bin, opt_ok, aco_ok)


# ─────────────────────────────────────────────────────────────────────────────
# REWARD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward(action, channel_state, transmission_result):
    """
    Multi-objective reward function.

    R = w1*throughput_norm + w2*energy_efficiency - w3*delay_norm
        + w4*delivery_success - w5*infeasibility_penalty

    Weights balance competing objectives as per the paper.
    """
    # Weights
    W_THROUGHPUT  = 0.35
    W_ENERGY      = 0.30
    W_DELAY       = 0.20
    W_DELIVERY    = 0.15

    tp      = transmission_result.get('throughput_norm', 0)
    ee      = transmission_result.get('energy_efficiency', 0)
    delay   = transmission_result.get('delay_norm', 0)
    pdr     = transmission_result.get('pdr', 0)
    penalty = transmission_result.get('infeasibility_penalty', 0)

    reward = (W_THROUGHPUT * tp
              + W_ENERGY   * ee
              - W_DELAY    * delay
              + W_DELIVERY * pdr
              - penalty)

    return float(np.clip(reward, -1, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Q-LEARNING AGENT
# ─────────────────────────────────────────────────────────────────────────────
class QLearningAgent:
    """
    Tabular Q-Learning for adaptive mode switching.

    Q(s,a) ← Q(s,a) + α * [r + γ*max_a' Q(s',a') - Q(s,a)]

    Hyperparameters:
      α (alpha)   = learning rate
      γ (gamma)   = discount factor
      ε (epsilon) = exploration rate (decays over time)
    """

    def __init__(self,
                 alpha=0.1,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.995):

        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: defaultdict so unseen states get 0
        self.Q = defaultdict(_zero_q_row)
        self.discretizer = StateDiscretizer()

        # Training history
        self.episode_rewards   = []
        self.episode_actions   = []
        self.q_value_history   = []
        self.epsilon_history   = []

    def get_state(self, channel_state):
        return self.discretizer.discretize(channel_state)

    def select_action(self, channel_state):
        """
        ε-greedy action selection.
        Infeasible actions are masked out before selection.
        """
        state = self.get_state(channel_state)

        # Build feasibility mask
        mask = np.ones(N_ACTIONS, dtype=bool)
        if not channel_state.optical_feasible():
            mask[ACTION_OPTICAL] = False
        if not channel_state.acoustic_feasible():
            mask[ACTION_ACOUSTIC] = False
        # Hybrid needs at least one feasible
        if not (channel_state.optical_feasible() or
                channel_state.acoustic_feasible()):
            mask[ACTION_HYBRID] = False

        # If nothing feasible, default to acoustic (longest range)
        if not np.any(mask):
            return ACTION_ACOUSTIC

        # ε-greedy
        if random.random() < self.epsilon:
            feasible_actions = list(np.where(mask)[0])
            return int(random.choice(feasible_actions))
        else:
            q_values = self.Q[state].copy()
            q_values[~mask] = -np.inf   # mask infeasible
            return int(np.argmax(q_values))

    def update(self, state_cur, action, reward, state_next, done=False):
        """Q-Learning update rule."""
        s  = self.get_state(state_cur)
        s_ = self.get_state(state_next)

        current_q  = self.Q[s][action]
        max_next_q = 0.0 if done else np.max(self.Q[s_])

        td_target  = reward + self.gamma * max_next_q
        td_error   = td_target - current_q
        self.Q[s][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Reduce exploration over time."""
        self.epsilon = max(self.epsilon_min,
                          self.epsilon * self.epsilon_decay)

    def get_best_action(self, channel_state):
        """Pure greedy action (no exploration) for evaluation."""
        state = self.get_state(channel_state)
        return int(np.argmax(self.Q[state]))

    def save_history(self, episode_reward, actions):
        self.episode_rewards.append(episode_reward)
        self.episode_actions.append(actions)
        self.epsilon_history.append(self.epsilon)
        # Average Q-value across all known states
        if self.Q:
            avg_q = np.mean([np.max(v) for v in self.Q.values()])
            self.q_value_history.append(avg_q)

    @property
    def n_states_visited(self):
        return len(self.Q)


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED BASELINE (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
class RuleBasedSwitching:
    """
    Traditional threshold-based switching (baseline comparison).
    No learning, no adaptation to energy/environment changes.
    """
    OPTICAL_MAX_DIST   = 80    # m
    OPTICAL_MAX_TURB   = 1.5   # turbidity threshold
    ENERGY_THRESHOLD   = 10    # J, below this use acoustic (lower power)

    def select_action(self, channel_state):
        d  = channel_state.distance
        e  = channel_state.energy
        t  = channel_state.turbidity

        if (d <= self.OPTICAL_MAX_DIST and
                t <= self.OPTICAL_MAX_TURB and
                e > self.ENERGY_THRESHOLD):
            return ACTION_OPTICAL
        elif d <= 100 and e > self.ENERGY_THRESHOLD:
            return ACTION_HYBRID
        else:
            return ACTION_ACOUSTIC


# ─────────────────────────────────────────────────────────────────────────────
# STATIC ACOUSTIC ONLY (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
class AcousticOnly:
    """Always uses acoustic — traditional single-mode baseline."""
    def select_action(self, channel_state):
        return ACTION_ACOUSTIC


# ─────────────────────────────────────────────────────────────────────────────
# NON-ADAPTIVE HYBRID (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
class NonAdaptiveHybrid:
    """Fixed 50/50 hybrid — does not adapt to conditions."""
    def select_action(self, channel_state):
        if channel_state.optical_feasible():
            return ACTION_HYBRID
        return ACTION_ACOUSTIC
