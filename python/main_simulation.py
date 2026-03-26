"""
=============================================================================
MAIN TRAINING + EVALUATION SCRIPT
=============================================================================
Trains Q-Learning agent and evaluates against 3 baselines:
  1. Acoustic Only (traditional)
  2. Rule-Based Hybrid (non-adaptive)
  3. Non-Adaptive Hybrid
  4. Q-Learning Adaptive Hybrid (proposed)

Collects all metrics and saves results for plotting.
=============================================================================
"""

import numpy as np
import random
import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(__file__))

from channel_models import ChannelState
from q_learning_agent import (QLearningAgent, RuleBasedSwitching,
                               AcousticOnly, NonAdaptiveHybrid,
                               ACTION_NAMES)
from environment import UIoTEnvironment

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SEED           = 42
N_NODES        = 20
TRAIN_EPISODES = 300
EVAL_EPISODES  = 50
MAX_STEPS      = 300
WATER_TYPE     = 'clear_ocean'

random.seed(SEED)
np.random.seed(SEED)

print("=" * 65)
print("  Hybrid Optical-Acoustic UIoT — Q-Learning Simulation")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_agent(agent, env, n_episodes):
    print(f"\n[TRAINING] {n_episodes} episodes × {MAX_STEPS} steps/episode")
    print(f"  Nodes={N_NODES}, Water={WATER_TYPE}")
    print("-" * 55)

    train_rewards   = []
    train_lifetime  = []
    train_pdr       = []
    train_energy    = []

    for ep in range(n_episodes):
        state, src  = env.reset()
        ep_reward   = 0
        ep_actions  = []

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            ep_actions.append(action)

            next_state, next_src, reward, done, info = env.step(
                action, state, src)

            agent.update(state, action, reward, next_state, done)

            ep_reward += reward
            state, src = next_state, next_src

            if done:
                break

        agent.decay_epsilon()
        summary = env.summary()

        train_rewards.append(ep_reward)
        train_lifetime.append(summary['steps'])
        train_pdr.append(summary['pdr'])
        train_energy.append(summary['total_energy_j'])
        agent.save_history(ep_reward, ep_actions)

        if (ep + 1) % 50 == 0:
            avg_r  = np.mean(train_rewards[-50:])
            avg_lt = np.mean(train_lifetime[-50:])
            print(f"  Ep {ep+1:4d}/{n_episodes} | "
                  f"AvgReward={avg_r:+.3f} | "
                  f"AvgLifetime={avg_lt:.0f} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"States={agent.n_states_visited}")

    print(f"\n  Training complete. States visited: {agent.n_states_visited}")
    return {
        'rewards':   train_rewards,
        'lifetime':  train_lifetime,
        'pdr':       train_pdr,
        'energy':    train_energy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_agent(agent_or_policy, env, n_episodes, label):
    print(f"\n[EVAL] {label} — {n_episodes} episodes")

    metrics = {
        'throughput':  [],
        'latency':     [],
        'pdr':         [],
        'energy':      [],
        'lifetime':    [],
        'mode_dist':   [],
    }

    use_ql = isinstance(agent_or_policy, QLearningAgent)

    for ep in range(n_episodes):
        state, src = env.reset()

        for step in range(MAX_STEPS):
            if use_ql:
                action = agent_or_policy.get_best_action(state)
            else:
                action = agent_or_policy.select_action(state)

            next_state, next_src, reward, done, info = env.step(
                action, state, src)
            state, src = next_state, next_src
            if done:
                break

        summary = env.summary()
        metrics['throughput'].append(summary['avg_throughput_bps'])
        metrics['latency'].append(summary['avg_latency_ms'])
        metrics['pdr'].append(summary['pdr'])
        metrics['energy'].append(summary['total_energy_j'])
        metrics['lifetime'].append(summary['steps'])
        metrics['mode_dist'].append(summary['mode_counts'].copy())

    # Aggregate
    agg = {k: np.mean(v) for k, v in metrics.items()
           if k != 'mode_dist'}
    agg['std_throughput'] = np.std(metrics['throughput'])
    agg['std_latency']    = np.std(metrics['latency'])
    agg['std_lifetime']   = np.std(metrics['lifetime'])
    agg['mode_dist_raw']  = metrics['mode_dist']
    agg['label']          = label
    agg['all_throughput'] = metrics['throughput']
    agg['all_latency']    = metrics['latency']
    agg['all_pdr']        = metrics['pdr']
    agg['all_energy']     = metrics['energy']
    agg['all_lifetime']   = metrics['lifetime']

    print(f"  Throughput : {agg['throughput']/1e6:8.2f} Mbps "
          f"(±{agg['std_throughput']/1e6:.2f})")
    print(f"  Latency    : {agg['latency']:8.2f} ms "
          f"(±{agg['std_latency']:.2f})")
    print(f"  PDR        : {agg['pdr']*100:8.2f} %")
    print(f"  Energy     : {agg['energy']:8.2f} J")
    print(f"  Lifetime   : {agg['lifetime']:8.1f} steps "
          f"(±{agg['std_lifetime']:.1f})")

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS — vary distance, turbidity, nodes
# ─────────────────────────────────────────────────────────────────────────────
def sensitivity_analysis(agent, water_type='clear_ocean'):
    """Generate metric curves vs key parameters."""
    print("\n[SENSITIVITY ANALYSIS]")

    from channel_models import OpticalChannel, AcousticChannel

    # ── BER vs Distance ──────────────────────────────────────────────────
    distances = np.linspace(1, 5000, 300)
    opt_ch    = OpticalChannel(water_type)
    aco_ch    = AcousticChannel()

    opt_ber  = [opt_ch.ber(d) for d in distances]
    aco_ber  = [aco_ch.ber(d) for d in distances]
    opt_rate = [opt_ch.achievable_data_rate(d) for d in distances]
    aco_rate = [aco_ch.achievable_data_rate(d) for d in distances]
    opt_lat  = [opt_ch.latency(d) for d in distances]
    aco_lat  = [aco_ch.latency(d) for d in distances]
    opt_loss = [opt_ch.attenuation_db(d) for d in distances]
    aco_loss = [aco_ch.path_loss_db(d) for d in distances]

    # ── Throughput vs Node Count ─────────────────────────────────────────
    node_counts = [5, 10, 15, 20, 25, 30, 35, 40]
    tp_ql, tp_ac, tp_rb, tp_na = [], [], [], []

    policies = {
        'Q-Learning': agent,
        'Acoustic':   AcousticOnly(),
        'RuleBased':  RuleBasedSwitching(),
        'NonAdaptive':NonAdaptiveHybrid(),
    }

    print("  Running node-count sweep...")
    for nc in node_counts:
        row = {}
        for name, pol in policies.items():
            env_tmp = UIoTEnvironment(n_nodes=nc, max_steps=150)
            res = evaluate_agent(pol, env_tmp, 10, f"{name}-{nc}nodes")
            row[name] = res['throughput'] / 1e6   # Mbps
        tp_ql.append(row['Q-Learning'])
        tp_ac.append(row['Acoustic'])
        tp_rb.append(row['RuleBased'])
        tp_na.append(row['NonAdaptive'])
        print(f"    Nodes={nc}: QL={row['Q-Learning']:.1f} "
              f"| AC={row['Acoustic']:.2f} "
              f"| RB={row['RuleBased']:.1f} Mbps")

    # ── Energy vs Time (episode steps) ──────────────────────────────────
    print("  Running energy-over-time sweep...")
    env_tmp = UIoTEnvironment(n_nodes=20, max_steps=300)
    energy_over_time = {k: [] for k in policies}
    alive_over_time  = {k: [] for k in policies}

    for name, pol in policies.items():
        env_tmp.reset()
        state, src = env_tmp.reset()
        use_ql = isinstance(pol, QLearningAgent)
        e_hist = []
        a_hist = []
        for step in range(300):
            if use_ql:
                action = pol.get_best_action(state)
            else:
                action = pol.select_action(state)
            next_state, next_src, _, done, _ = env_tmp.step(
                action, state, src)
            e_hist.append(env_tmp.avg_residual_energy)
            a_hist.append(env_tmp.alive_ratio * 100)
            state, src = next_state, next_src
            if done:
                break
        energy_over_time[name] = e_hist
        alive_over_time[name]  = a_hist

    return {
        'distances':        distances,
        'opt_ber':          opt_ber,
        'aco_ber':          aco_ber,
        'opt_rate':         opt_rate,
        'aco_rate':         aco_rate,
        'opt_latency':      opt_lat,
        'aco_latency':      aco_lat,
        'opt_loss':         opt_loss,
        'aco_loss':         aco_loss,
        'node_counts':      node_counts,
        'tp_ql':            tp_ql,
        'tp_ac':            tp_ac,
        'tp_rb':            tp_rb,
        'tp_na':            tp_na,
        'energy_over_time': energy_over_time,
        'alive_over_time':  alive_over_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # 1. Create environment & agent
    env   = UIoTEnvironment(n_nodes=N_NODES, max_steps=MAX_STEPS,
                             water_type=WATER_TYPE)
    agent = QLearningAgent(alpha=0.1, gamma=0.95,
                           epsilon=1.0, epsilon_min=0.05,
                           epsilon_decay=0.992)

    # 2. Train
    train_history = train_agent(agent, env, TRAIN_EPISODES)

    # 3. Evaluate all policies
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)

    eval_env = UIoTEnvironment(n_nodes=N_NODES, max_steps=MAX_STEPS,
                                water_type=WATER_TYPE)

    results = {}
    results['Q-Learning (Proposed)'] = evaluate_agent(
        agent, eval_env, EVAL_EPISODES, 'Q-Learning (Proposed)')
    results['Acoustic Only'] = evaluate_agent(
        AcousticOnly(), eval_env, EVAL_EPISODES, 'Acoustic Only')
    results['Rule-Based Hybrid'] = evaluate_agent(
        RuleBasedSwitching(), eval_env, EVAL_EPISODES, 'Rule-Based Hybrid')
    results['Non-Adaptive Hybrid'] = evaluate_agent(
        NonAdaptiveHybrid(), eval_env, EVAL_EPISODES, 'Non-Adaptive Hybrid')

    # 4. Sensitivity analysis
    sens_data = sensitivity_analysis(agent, WATER_TYPE)

    # 5. Save everything
    output = {
        'train_history': train_history,
        'eval_results':  results,
        'sens_data':     sens_data,
        'agent':         agent,
    }
    with open('simulation_results.pkl', 'wb') as f:
        pickle.dump(output, f)

    print("\n[✓] All results saved to simulation_results.pkl")
    print("[✓] Run plot_results.py to generate all 8 performance graphs")
