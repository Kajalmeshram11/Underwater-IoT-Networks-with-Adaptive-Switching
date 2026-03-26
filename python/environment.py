"""
=============================================================================
UNDERWATER IoT NETWORK SIMULATION ENVIRONMENT
=============================================================================
Simulates a UIoT network with N sensor nodes deployed underwater.
Each node has:
  - Position (x, y, z)
  - Residual energy
  - Optical + acoustic transceivers
  - Data queue

The environment drives the Q-Learning agent's training loop.
=============================================================================
"""

import numpy as np
import random
from channel_models import (OpticalChannel, AcousticChannel,
                             ChannelState, WATER_TYPES)
from q_learning_agent import (ACTION_OPTICAL, ACTION_ACOUSTIC, ACTION_HYBRID,
                               ACTION_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR NODE
# ─────────────────────────────────────────────────────────────────────────────
class UnderwaterNode:
    """
    Single underwater sensor node.
    """
    INITIAL_ENERGY_J = 100.0    # Joules
    IDLE_POWER_W     = 0.01     # Watts when idle
    SWITCHING_COST_J = 0.001    # Energy cost of mode switching

    def __init__(self, node_id, x, y, z, water_type='clear_ocean'):
        self.id             = node_id
        self.position       = np.array([x, y, z], dtype=float)
        self.energy         = self.INITIAL_ENERGY_J
        self.water_type     = water_type
        self.optical_ch     = OpticalChannel(water_type)
        self.acoustic_ch    = AcousticChannel()
        self.packets_sent   = 0
        self.packets_failed = 0
        self.last_mode      = None
        self.energy_history = [self.energy]
        self.alive          = True

    def distance_to(self, other_node):
        return float(np.linalg.norm(self.position - other_node.position))

    def deplete_energy(self, amount_j):
        self.energy -= amount_j
        if self.energy <= 0:
            self.energy = 0
            self.alive  = False
        self.energy_history.append(self.energy)

    def idle_consumption(self, time_s=1.0):
        self.deplete_energy(self.IDLE_POWER_W * time_s)

    def switching_overhead(self, new_mode):
        """Pay switching cost if mode changes."""
        if self.last_mode is not None and self.last_mode != new_mode:
            self.deplete_energy(self.SWITCHING_COST_J)
        self.last_mode = new_mode

    @property
    def pdr(self):
        total = self.packets_sent + self.packets_failed
        return self.packets_sent / total if total > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TRANSMISSION SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class TransmissionSimulator:
    """
    Simulates a single packet transmission between two nodes
    under a given action (optical / acoustic / hybrid).
    Returns normalized metrics for reward computation.
    """

    PACKET_SIZE_BITS = 1024 * 8   # 8 KB packet

    def simulate(self, src_node, dst_node, action,
                 data_size_kb, turbidity_factor, water_type):
        """
        Returns dict with transmission results + reward components.
        """
        d  = src_node.distance_to(dst_node)
        E  = src_node.energy
        opt_ch = src_node.optical_ch
        aco_ch = src_node.acoustic_ch

        result = {
            'throughput_bps':         0,
            'latency_ms':             0,
            'energy_used_j':          0,
            'pdr':                    0,
            'action_taken':           action,
            'throughput_norm':        0,
            'energy_efficiency':      0,
            'delay_norm':             0,
            'infeasibility_penalty':  0,
            'success':                False,
        }

        # ── OPTICAL ──────────────────────────────────────────────────────────
        if action == ACTION_OPTICAL:
            if not opt_ch.is_feasible(d, turbidity_factor):
                result['infeasibility_penalty'] = 1.0
                result['pdr'] = 0
                src_node.packets_failed += 1
                return result

            rate    = opt_ch.achievable_data_rate(d)
            e_bit   = opt_ch.energy_per_bit(d)
            latency = opt_ch.latency(d)
            e_total = e_bit * data_size_kb * 1024 * 8
            ber     = opt_ch.ber(d)
            pdr     = max(0, 1.0 - ber * self.PACKET_SIZE_BITS)

            result.update({
                'throughput_bps':    rate,
                'latency_ms':        latency,
                'energy_used_j':     e_total,
                'pdr':               pdr,
                'success':           pdr > 0.8,
                'throughput_norm':   min(rate / 10e9, 1.0),
                'energy_efficiency': min(1.0 / (e_bit * 1e6 + 1e-9), 1.0),
                'delay_norm':        min(latency / 1000, 1.0),
            })

        # ── ACOUSTIC ─────────────────────────────────────────────────────────
        elif action == ACTION_ACOUSTIC:
            if not aco_ch.is_feasible(d):
                result['infeasibility_penalty'] = 1.0
                result['pdr'] = 0
                src_node.packets_failed += 1
                return result

            rate    = aco_ch.achievable_data_rate(d)
            e_bit   = aco_ch.energy_per_bit(d)
            latency = aco_ch.latency(d)
            e_total = e_bit * data_size_kb * 1024 * 8
            ber     = aco_ch.ber(d)
            pdr     = max(0, 1.0 - ber * self.PACKET_SIZE_BITS)

            result.update({
                'throughput_bps':    rate,
                'latency_ms':        latency,
                'energy_used_j':     e_total,
                'pdr':               pdr,
                'success':           pdr > 0.6,
                'throughput_norm':   min(rate / 100e3, 1.0),
                'energy_efficiency': min(1.0 / (e_bit * 1e3 + 1e-9), 1.0),
                'delay_norm':        min(latency / 5000, 1.0),
            })

        # ── HYBRID ───────────────────────────────────────────────────────────
        elif action == ACTION_HYBRID:
            opt_ok = opt_ch.is_feasible(d, turbidity_factor)
            aco_ok = aco_ch.is_feasible(d)

            if not (opt_ok or aco_ok):
                result['infeasibility_penalty'] = 1.0
                result['pdr'] = 0
                src_node.packets_failed += 1
                return result

            # Route high-priority data over optical, rest over acoustic
            if opt_ok and aco_ok:
                opt_share = 0.7   # 70% over optical
                aco_share = 0.3
            elif opt_ok:
                opt_share = 1.0
                aco_share = 0.0
            else:
                opt_share = 0.0
                aco_share = 1.0

            opt_rate  = opt_ch.achievable_data_rate(d) * opt_share
            aco_rate  = aco_ch.achievable_data_rate(d) * aco_share
            total_rate= opt_rate + aco_rate

            opt_e     = opt_ch.energy_per_bit(d) * opt_share if opt_ok else 0
            aco_e     = aco_ch.energy_per_bit(d) * aco_share
            avg_e_bit = (opt_e + aco_e)

            opt_lat   = opt_ch.latency(d)
            aco_lat   = aco_ch.latency(d)
            avg_lat   = (opt_lat * opt_share + aco_lat * aco_share)

            opt_ber   = opt_ch.ber(d) if opt_ok else 1.0
            aco_ber   = aco_ch.ber(d) if aco_ok else 1.0
            avg_ber   = opt_ber * opt_share + aco_ber * aco_share
            pdr       = max(0, 1.0 - avg_ber * self.PACKET_SIZE_BITS)

            e_total   = avg_e_bit * data_size_kb * 1024 * 8

            result.update({
                'throughput_bps':    total_rate,
                'latency_ms':        avg_lat,
                'energy_used_j':     e_total,
                'pdr':               pdr,
                'success':           pdr > 0.75,
                'throughput_norm':   min(total_rate / 7e9, 1.0),
                'energy_efficiency': min(1.0 / (avg_e_bit * 1e5 + 1e-9), 1.0),
                'delay_norm':        min(avg_lat / 3000, 1.0),
            })

        # Deduct energy
        e_used = result['energy_used_j']
        src_node.deplete_energy(e_used)
        src_node.switching_overhead(action)

        if result['success']:
            src_node.packets_sent += 1
        else:
            src_node.packets_failed += 1

        return result


# ─────────────────────────────────────────────────────────────────────────────
# UIoT ENVIRONMENT (Gym-style)
# ─────────────────────────────────────────────────────────────────────────────
class UIoTEnvironment:
    """
    UIoT simulation environment.
    Drives Q-Learning training and performance evaluation.

    Episode = one "life" of the network until all nodes die or
              max_steps reached.
    """

    def __init__(self,
                 n_nodes=20,
                 area_m=1000,
                 depth_m=200,
                 water_type='clear_ocean',
                 max_steps=500):

        self.n_nodes    = n_nodes
        self.area_m     = area_m
        self.depth_m    = depth_m
        self.water_type = water_type
        self.max_steps  = max_steps
        self.simulator  = TransmissionSimulator()

        # Metrics tracking
        self.reset()

    def reset(self):
        """Initialize / reset the network."""
        # Deploy nodes randomly in 3D space
        self.nodes = []
        for i in range(self.n_nodes):
            x = random.uniform(0, self.area_m)
            y = random.uniform(0, self.area_m)
            z = random.uniform(0, self.depth_m)
            self.nodes.append(
                UnderwaterNode(i, x, y, z, self.water_type))

        # Sink node at surface centre
        self.sink = UnderwaterNode(
            999, self.area_m / 2, self.area_m / 2, 0,
            self.water_type)
        self.sink.energy = float('inf')   # sink has unlimited power

        self.step_count      = 0
        self.total_throughput= 0
        self.total_energy    = 0
        self.total_delay     = 0
        self.total_packets   = 0
        self.mode_counts     = {0: 0, 1: 0, 2: 0}

        return self._get_random_state()

    def _get_random_state(self):
        """Sample a random channel state from current network."""
        alive_nodes = [n for n in self.nodes if n.alive]
        if not alive_nodes:
            src = self.nodes[0]
        else:
            src = random.choice(alive_nodes)

        d           = src.distance_to(self.sink)
        turbidity   = random.uniform(0.5, 3.0)
        data_size   = random.uniform(10, 500)
        state       = ChannelState(d, src.energy, data_size,
                                   turbidity, self.water_type)
        return state, src

    def step(self, action, channel_state, src_node):
        """
        Execute one transmission step.
        Returns: (next_state, src_node, reward, done, info)
        """
        # Simulate transmission
        result = self.simulator.simulate(
            src_node, self.sink, action,
            channel_state.data_size,
            channel_state.turbidity,
            self.water_type)

        # Idle consumption for all nodes
        for n in self.nodes:
            if n.alive:
                n.idle_consumption(0.1)

        # Track metrics
        self.total_throughput += result['throughput_bps']
        self.total_energy     += result['energy_used_j']
        self.total_delay      += result['latency_ms']
        self.total_packets    += 1
        self.mode_counts[action] += 1
        self.step_count += 1

        # Compute reward
        from q_learning_agent import compute_reward
        reward = compute_reward(action, channel_state, result)

        # Check termination
        alive = sum(1 for n in self.nodes if n.alive)
        done  = (alive == 0 or self.step_count >= self.max_steps)

        # Next state
        next_state, next_src = self._get_random_state()

        info = {
            'alive_nodes':    alive,
            'throughput_bps': result['throughput_bps'],
            'latency_ms':     result['latency_ms'],
            'energy_used_j':  result['energy_used_j'],
            'pdr':            result['pdr'],
            'success':        result['success'],
            'action_name':    ACTION_NAMES[action],
            'step':           self.step_count,
        }

        return next_state, next_src, reward, done, info

    @property
    def alive_ratio(self):
        return sum(1 for n in self.nodes if n.alive) / self.n_nodes

    @property
    def avg_residual_energy(self):
        alive = [n.energy for n in self.nodes if n.alive]
        return np.mean(alive) if alive else 0

    @property
    def network_lifetime_steps(self):
        return self.step_count

    def summary(self):
        avg_tp  = self.total_throughput / max(self.total_packets, 1)
        avg_lat = self.total_delay      / max(self.total_packets, 1)
        pdr_all = sum(n.packets_sent for n in self.nodes)
        pdr_tot = pdr_all + sum(n.packets_failed for n in self.nodes)
        pdr     = pdr_all / max(pdr_tot, 1)

        return {
            'steps':              self.step_count,
            'avg_throughput_bps': avg_tp,
            'avg_latency_ms':     avg_lat,
            'pdr':                pdr,
            'total_energy_j':     self.total_energy,
            'alive_ratio':        self.alive_ratio,
            'mode_counts':        self.mode_counts,
        }
