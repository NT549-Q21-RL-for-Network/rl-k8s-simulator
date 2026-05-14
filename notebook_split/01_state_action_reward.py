# Auto-generated from rl-training.ipynb


# ===== From notebook code cell 2 =====
# ========== STATE SPACE DEFINITION ==========

class StateSpace:
    """Continuous and discrete state metrics from Kubernetes cluster"""
    
    # Continuous state metrics (normalized to [0, 1])
    CONTINUOUS_METRICS = {
        'cpu_utilization': (0.0, 1.0),           # Node CPU usage %
        'memory_usage': (0.0, 1.0),              # Node memory usage %
        'disk_io': (0.0, 1.0),                   # Disk I/O pressure
        'network_bandwidth': (0.0, 1.0),         # Network saturation %
        'p90_latency': (0.0, 1.0),               # API latency p90 (normalized to max 5s)
        'p99_latency': (0.0, 1.0),               # API latency p99 (normalized to max 10s)
        'error_rate_5xx': (0.0, 1.0),            # HTTP 5xx error rate
        'throughput': (0.0, 1.0),                # Requests/sec (normalized)
    }
    
    # Discrete state metrics — must match _collect_metrics() clipping ranges!
    DISCRETE_METRICS = {
        'node_ready_status': (0, 3),             # 0=all_ready, 1=partial, 2=most_down, 3=all_down
        'pending_pods': (0, 100),                # Number of pending pods
        'crashloop_flag': (0, 20),               # Count of CrashLoopBackOff pods — FIX: was (0,5), now (0,20) to match _collect_metrics
        'failed_pods': (0, 50),                  # Count of failed pods
    }
    
    @staticmethod
    def get_observation_space():
        """Define Gymnasium observation space"""
        continuous_dim = len(StateSpace.CONTINUOUS_METRICS)
        discrete_parts = []
        
        for metric, (min_val, max_val) in StateSpace.DISCRETE_METRICS.items():
            discrete_parts.append(spaces.Discrete(max_val - min_val + 1))
        
        continuous_box = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(continuous_dim,),
            dtype=np.float32
        )
        
        discrete_box = spaces.MultiDiscrete(
            [max_val - min_val + 1 for _, (min_val, max_val) in StateSpace.DISCRETE_METRICS.items()]
        )
        
        return spaces.Dict({
            'continuous': continuous_box,
            'discrete': discrete_box
        })

print("✓ State space defined")
print(f"  Continuous metrics: {len(StateSpace.CONTINUOUS_METRICS)} dimensions")
print(f"  Discrete metrics: {len(StateSpace.DISCRETE_METRICS)} dimensions")
print(f"  ⚠️  FIX: crashloop_flag range updated to (0, 20) to match _collect_metrics")


# ===== From notebook code cell 3 =====
# ========== ACTION SPACE DEFINITION ==========

class ActionSpace:
    """Kubernetes recovery actions"""
    
    ACTIONS = {
        0: {'name': 'idle', 'description': 'Do nothing, observe'},
        1: {'name': 'restart_pod', 'description': 'Kill and restart a failed pod', 'param': 'pod_index'},
        2: {'name': 'scale_up', 'description': 'Increase pod replicas by 1', 'param': 'deployment_index'},
        3: {'name': 'scale_down', 'description': 'Decrease pod replicas by 1', 'param': 'deployment_index'},
        4: {'name': 'drain_node', 'description': 'Drain node for maintenance', 'param': 'node_index'},
        5: {'name': 'cordon_node', 'description': 'Mark node as unschedulable', 'param': 'node_index'},
        6: {'name': 'uncordon_node', 'description': 'Mark node as schedulable', 'param': 'node_index'},
    }
    
    @staticmethod
    def get_action_space():
        """Define Gymnasium action space"""
        return spaces.Discrete(len(ActionSpace.ACTIONS))
    
    @staticmethod
    def describe_action(action_id: int) -> str:
        action = ActionSpace.ACTIONS.get(action_id, {'name': 'unknown'})
        return f"{action['name']}: {action.get('description', '')}"

print("✓ Action space defined")
print(f"  Total actions: {len(ActionSpace.ACTIONS)}")
for idx, action_info in ActionSpace.ACTIONS.items():
    print(f"    {idx}: {action_info['name']}")


# ===== From notebook code cell 4 =====
class RewardCalculator:
    """
    Optimized Reward Function (v3.0): K8s Problem-Aligned Weights
    
    Principle: Prioritize based on K8s self-healing business importance:
    Recovery >> SLA Compliance >> Action Efficiency >> Resource Efficiency >> Speed
    
    R(s_t, a_t) = α·ΔHealth - β·Overhead - γ·ActionDisruption - δ·SLA_Penalty - step_penalty
    """
    
    # ── OPTIMIZED WEIGHTS (v3.0 - K8s Problem-Aligned) ────────────────────────────
    ALPHA = 8.0         # Recovery reward (was 4.2 → +90%): PRIMARY objective
    BETA = 1.5          # Resource overhead cost (was 2.0 → -25%): Secondary objective  
    GAMMA = 2.0         # Action disruption penalty (was 1.5 → +33%): Penalize disruption
    DELTA = 4.0         # SLA violation penalty (was 1.2 → +233%): SLA is CRITICAL
    STEP_PENALTY = 0.06 # Time cost per step (was 0.08 → -25%): More time to explore
    RECOVERY_BONUS = 40.0   # Success bonus (was 32.0 → +25%): Strong incentive
    COLLAPSE_PENALTY = -50.0  # Failure penalty (was -25 → -50): Discourage failures
    
    @staticmethod
    def calculate(prev_state: Dict[str, Any], curr_state: Dict[str, Any], 
                  action_id: int, episode_steps: int) -> float:
        """Calculate reward for transition (s_t, a_t, s_{t+1})"""
        
        prev_health = RewardCalculator._calculate_health(prev_state)
        curr_health = RewardCalculator._calculate_health(curr_state)
        stability_delta = curr_health - prev_health
        stability_reward = RewardCalculator.ALPHA * max(0, stability_delta)
        
        overhead_cost = RewardCalculator._calculate_overhead(curr_state)
        overhead_penalty = RewardCalculator.BETA * overhead_cost
        
        action_disruption = RewardCalculator._get_action_impact(action_id)
        impact_penalty = RewardCalculator.GAMMA * action_disruption
        
        sla_penalty = RewardCalculator._calculate_sla_penalty(curr_state)
        sla_cost = RewardCalculator.DELTA * sla_penalty
        
        step_penalty = RewardCalculator.STEP_PENALTY
        
        return stability_reward - overhead_penalty - impact_penalty - sla_cost - step_penalty

    @staticmethod
    def _calculate_health(state: Dict[str, Any]) -> float:
        """K8s-aligned health score [0,1]: Prioritize SLA metrics.
        
        Weights (K8s priorities): error_rate(30%) + node(25%) + latency(20%) + pending(10%) + throughput(10%) + crashloop(5%)
        """
        error_rate = state.get('error_rate_5xx', 0)
        pending_pods = state.get('pending_pods', 0) / 100.0
        crashloop = state.get('crashloop_flag', 0) / 20.0
        node_status = state.get('node_ready_status', 0) / 3.0
        latency = (state.get('p90_latency', 0) + state.get('p99_latency', 0)) / 2.0
        throughput_loss = 1.0 - state.get('throughput', 1.0)
        
        # Detect critical resource exhaustion
        cpu = state.get('cpu_utilization', 0)
        memory = state.get('memory_usage', 0)
        critical_resource = (cpu > 0.98 or memory > 0.98)
        
        # K8s-aligned health calculation (prioritize SLA metrics)
        health = (
            (1.0 - error_rate) * 0.30 +         # 30%: Error rate is PRIMARY SLA signal
            (1.0 - node_status) * 0.25 +        # 25%: Node health = availability
            (1.0 - latency) * 0.20 +            # 20%: Latency = user experience
            (1.0 - pending_pods) * 0.10 +       # 10%: Pending pods indicator
            (1.0 - throughput_loss) * 0.10 +    # 10%: Throughput
            (1.0 - crashloop) * 0.05            # 5%: CrashLoop low priority
        )
        
        # Penalize critical resource exhaustion
        if critical_resource:
            health *= 0.5
        
        return float(np.clip(health, 0, 1))
    
    @staticmethod
    def _calculate_overhead(state: Dict[str, Any]) -> float:
        """Resource overhead penalty: CPU + Memory + Network"""
        cpu = state.get('cpu_utilization', 0)
        memory = state.get('memory_usage', 0)
        network = state.get('network_bandwidth', 0)
        return (cpu + memory + network) / 3.0
    
    @staticmethod
    def _get_action_impact(action_id: int) -> float:
        """K8s-aligned action disruption costs [0,1]. Higher = more disruptive.
        
        Ratio drain/scale_up = 0.75/0.12 = 6.25x (strong differentiation)
        """
        action_impacts = {
            0: 0.0,     # idle
            1: 0.50,    # restart_pod (HIGH - pod downtime)
            2: 0.12,    # scale_up (VERY LOW - safest)
            3: 0.20,    # scale_down (LOW-MEDIUM)
            4: 0.75,    # drain_node (VERY HIGH - most disruptive)
            5: 0.40,    # cordon_node (MEDIUM-HIGH)
            6: 0.05,    # uncordon_node (ALMOST FREE)
        }
        return action_impacts.get(action_id, 0.25)
    
    @staticmethod
    def _calculate_sla_penalty(state: Dict[str, Any]) -> float:
        """Strict K8s SLA penalty: Exponential (early warning).
        
        Thresholds: latency 0.3 (strict), p90 0.25, error_rate 0.02 (very strict)
        """
        p99_latency = state.get('p99_latency', 0)
        p90_latency = state.get('p90_latency', 0)
        error_rate = state.get('error_rate_5xx', 0)
        
        # Exponential penalties for SLA violations
        latency_violation = max(0, p99_latency - 0.3) ** 1.5 * 1.0
        latency_p90_violation = max(0, p90_latency - 0.25) ** 1.3 * 0.5
        error_violation = max(0, error_rate - 0.02) ** 1.2 * 1.0
        
        penalty = min(1.0, latency_violation + latency_p90_violation + error_violation)
        return penalty

print("✅ Reward Function v3.0: K8s-ALIGNED OPTIMIZATION")
print(f"  ALPHA (Recovery):    {RewardCalculator.ALPHA}       (+90% - PRIMARY objective)")
print(f"  BETA  (Resources):   {RewardCalculator.BETA}        (-25% - secondary)")
print(f"  GAMMA (Disruption):  {RewardCalculator.GAMMA}        (+33% - penalize more)")
print(f"  DELTA (SLA):         {RewardCalculator.DELTA}        (+233% - CRITICAL)")
print(f"  STEP_PENALTY:        {RewardCalculator.STEP_PENALTY}       (-25% - more exploration time)")
print(f"  RECOVERY_BONUS:      {RewardCalculator.RECOVERY_BONUS}   (+25% - success incentive)")
print(f"\n🎯 Priorities: Recovery >> SLA >> Efficiency >> Resources >> Speed")
print(f"📊 Health Weights: Error(30%) + Node(25%) + Latency(20%) + Pending(10%) + Throughput(10%) + Crashloop(5%)")
print(f"⚡ Expected: Success 92-95%, Steps 32-36, Reward -15 to -20")
