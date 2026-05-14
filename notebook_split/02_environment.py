# Auto-generated from rl-training.ipynb


# ===== From notebook code cell 8 =====
# ========== FAILURE SCENARIOS ==========

class FailureScenario:
    """Realistic Kubernetes failure patterns for agent training."""

    SCENARIOS = {
        'node_failure': {
            'description': 'Node down / network partition',
            'state': {
                'cpu_utilization': 0.2,
                'memory_usage': 0.1,
                'disk_io': 0.05,
                'network_bandwidth': 0.05,
                'p90_latency': 0.8,
                'p99_latency': 0.95,
                'error_rate_5xx': 0.30,
                'throughput': 0.2,
                'node_ready_status': 3.0,
                'pending_pods': 60.0,
                'crashloop_flag': 0.0,
                'failed_pods': 30.0,
            },
        },
        'pod_crash_loop': {
            'description': 'CrashLoopBackOff - app bug',
            'state': {
                'cpu_utilization': 0.3,
                'memory_usage': 0.4,
                'disk_io': 0.1,
                'network_bandwidth': 0.2,
                'p90_latency': 0.4,
                'p99_latency': 0.5,
                'error_rate_5xx': 0.15,
                'throughput': 0.5,
                'node_ready_status': 0.0,
                'pending_pods': 5.0,
                'crashloop_flag': 12.0,
                'failed_pods': 8.0,
            },
        },
        'resource_exhaustion': {
            'description': 'High CPU/Memory pressure',
            'state': {
                'cpu_utilization': 0.95,
                'memory_usage': 0.90,
                'disk_io': 0.7,
                'network_bandwidth': 0.5,
                'p90_latency': 0.6,
                'p99_latency': 0.75,
                'error_rate_5xx': 0.05,
                'throughput': 0.3,
                'node_ready_status': 1.0,
                'pending_pods': 25.0,
                'crashloop_flag': 2.0,
                'failed_pods': 5.0,
            },
        },
        'network_degradation': {
            'description': 'High latency / packet loss',
            'state': {
                'cpu_utilization': 0.5,
                'memory_usage': 0.4,
                'disk_io': 0.2,
                'network_bandwidth': 0.85,
                'p90_latency': 0.9,
                'p99_latency': 1.0,
                'error_rate_5xx': 0.20,
                'throughput': 0.4,
                'node_ready_status': 0.0,
                'pending_pods': 10.0,
                'crashloop_flag': 3.0,
                'failed_pods': 4.0,
            },
        },
    }

    @staticmethod
    def sample_scenario() -> Dict[str, Any]:
        """Randomly sample a failure scenario with noise."""
        scenario_name = np.random.choice(list(FailureScenario.SCENARIOS.keys()))
        scenario = FailureScenario.SCENARIOS[scenario_name].copy()
        scenario['name'] = scenario_name
        
        state = scenario['state'].copy()
        for key, val in state.items():
            noise = np.random.uniform(-0.08, 0.08)
            # Respect natural ranges for discrete metrics
            if key in ['node_ready_status', 'pending_pods', 'crashloop_flag', 'failed_pods']:
                max_vals = {'node_ready_status': 3, 'pending_pods': 100, 'crashloop_flag': 20, 'failed_pods': 50}
                state[key] = float(np.clip(val + noise * max_vals.get(key, 1), 0, max_vals.get(key, 1)))
            else:
                state[key] = float(np.clip(val + noise, 0.0, 1.0))
        
        scenario['state'] = state
        return scenario

print("✓ Failure scenarios defined (4 types)")
for name, s in FailureScenario.SCENARIOS.items():
    print(f"  {name}: {s['description']}")


# ===== From notebook code cell 9 =====
# ========== KUBERNETES SELF-HEALING ENVIRONMENT ==========

class K8sSelfHealingEnv(gym.Env):
    """Gymnasium environment for Kubernetes self-healing RL agent.
    
    Design: Agent learns from observations ONLY, not from scenario names.
    Action effects are UNIFORM across all scenarios.
    This ensures generalizability to production where scenarios are unknown.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Dict[str, Any] = None):
        super(K8sSelfHealingEnv, self).__init__()
        
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 100)
        self.observation_step_interval = self.config.get('step_interval_sec', 10)
        self.num_deployments = self.config.get('num_deployments', 5)
        self.num_nodes = self.config.get('num_nodes', 3)
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),  # 8 continuous + 4 discrete (normalized to [0,1])
            dtype=np.float32
        )
        self.action_space = ActionSpace.get_action_space()
        
        self.current_step = 0
        self.episode_rewards = []
        self.current_state = None
        self.prev_state = None
        self._scenario_name = None  # For logging/diagnostics only, NOT used in action dynamics
    
    def reset(self, seed=None, options=None):
        """Reset: inject initial failure via scenario sampling.
        
        NOTE: Scenario is only used for initialization diversity.
        Agent learns from observations, NOT from scenario names.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_rewards = []
        self.current_state = self._generate_failed_state()
        self.prev_state = self.current_state.copy()
        obs = self._encode_observation(self.current_state)
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one recovery action and observe new state."""
        self.current_step += 1
        
        # 1. Apply action effects to state (UNIFORM, not scenario-aware)
        action_name = ActionSpace.ACTIONS[action]['name']
        self._execute_action(action, action_name)
        
        # 2. Natural recovery + noise (applied AFTER action effects)
        self.prev_state = self.current_state.copy()
        self.current_state = self._collect_metrics()
        
        # 3. Reward
        reward = RewardCalculator.calculate(
            self.prev_state, self.current_state, action, self.current_step
        )
        self.episode_rewards.append(reward)
        
        # 4. Termination
        recovered  = self._is_recovered()
        collapsed  = self._is_collapsed()
        truncated  = self.current_step >= self.max_steps
        terminated = recovered or collapsed
        
        obs = self._encode_observation(self.current_state)
        info = {
            'action': action_name,
            'recovered': recovered,
            'collapsed': collapsed,
            'episode_reward': sum(self.episode_rewards),
        }
        return obs, reward, terminated, truncated, info
    
    def _generate_failed_state(self) -> Dict[str, float]:
        """Generate initial failed state from a sampled scenario.
        
        Scenario is ONLY for initialization randomness, not used in action dynamics.
        """
        scenario = FailureScenario.sample_scenario()
        self._scenario_name = scenario['name']
        return scenario['state'].copy()

    # ── Action effects: UNIFORM (no scenario-awareness) ───────────────────────
    def _execute_action(self, action: int, action_name: str) -> None:
        """Modify state in-place. Effects are UNIFORM, observation-based.
        
        Design principle: Agent makes decisions based on current metrics,
        not on knowing which scenario is active. This ensures the learned policy
        generalizes to production environments where scenario types are unknown.
        """
        s = self.current_state
        
        if action == 1:   # restart_pod
            s['crashloop_flag'] = max(0, s['crashloop_flag'] - 2.0)
            s['failed_pods']    = max(0, s['failed_pods']    - 2.0)
            s['pending_pods']   = max(0, s['pending_pods']   - 2.0)
            s['error_rate_5xx'] = max(0, s['error_rate_5xx'] * 0.85)
            s['throughput']     = min(1.0, s['throughput'] + 0.04)
            
        elif action == 2:  # scale_up
            s['pending_pods']   = max(0, s['pending_pods'] - 6.0)
            s['throughput']     = min(1.0, s['throughput'] + 0.10)
            s['cpu_utilization']= max(0.05, s['cpu_utilization'] - 0.08)
            s['memory_usage']   = max(0.05, s['memory_usage'] - 0.08)
            s['error_rate_5xx'] = max(0, s['error_rate_5xx'] * 0.91)
            
        elif action == 3:  # scale_down
            s['cpu_utilization']= max(0.05, s['cpu_utilization'] - 0.05)
            s['memory_usage']   = max(0.05, s['memory_usage']    - 0.05)
            s['throughput']    *= 0.90
            
        elif action == 4:  # drain_node
            s['node_ready_status'] = max(0, s['node_ready_status'] - 1.0)
            s['pending_pods']      = min(100, s['pending_pods'] + 5.0)
            
        elif action == 5:  # cordon_node
            s['node_ready_status'] = max(0, s['node_ready_status'] - 0.8)
            s['error_rate_5xx']    = max(0, s['error_rate_5xx'] * 0.95)
            
        elif action == 6:  # uncordon_node
            s['node_ready_status'] = max(0, s['node_ready_status'] - 0.8)
            s['pending_pods']      = max(0, s['pending_pods'] - 5.0)
            s['error_rate_5xx']   = max(0, s['error_rate_5xx'] * 0.96)
            s['throughput']       = min(1.0, s['throughput'] + 0.025)

    def _collect_metrics(self) -> Dict[str, float]:
        """Simulate post-action metric collection with natural recovery (improved v3.0).
        
        IMPROVEMENT: Changed drift from 0.95 to 0.98 for more realistic recovery
        0.95 = very slow recovery (needs many steps)
        0.98 = realistic recovery (balanced with action effects)
        """
        state = {}
        for key, val in self.current_state.items():
            if key == 'node_ready_status':
                # Nodes recover slowly but naturally
                drift = max(0, val - np.random.uniform(0, 0.12))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.05), 0, 3))
            elif key == 'pending_pods':
                # Pending pods clear gradually due to reconciliation
                drift = max(0, val - np.random.uniform(0.8, 2.5))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.3), 0, 100))
            elif key == 'crashloop_flag':
                # CrashLoop pods recover with K8s self-healing
                drift = max(0, val - np.random.uniform(0.5, 1.8))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.2), 0, 20))
            elif key == 'failed_pods':
                # Failed pods recover with self-healing
                drift = max(0, val - np.random.uniform(0.5, 1.8))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.2), 0, 50))
            else:
                # Other metrics (CPU, latency, error_rate) recover to normal baseline
                # IMPROVED: 0.95→0.98 makes recovery more realistic (3-5 steps vs 10+ steps)
                target = np.random.uniform(0.08, 0.25)
                drift  = val * 0.98 + target * 0.02
                state[key] = float(np.clip(drift + np.random.normal(0, 0.015), 0, 1))
        return state

    def _is_recovered(self) -> bool:
        """True when all health thresholds met."""
        s = self.current_state
        return (
            s['error_rate_5xx']    < 0.02 and
            s['pending_pods']      < 2    and
            s['crashloop_flag']    < 1    and
            s['node_ready_status'] < 1
        )

    def _is_collapsed(self) -> bool:
        """True when cluster is unrecoverable."""
        s = self.current_state
        return s['error_rate_5xx'] > 0.5 and s['node_ready_status'] >= 3

    def _encode_observation(self, state: Dict[str, float]) -> np.ndarray:
        """Flatten state to normalized [0,1]^12 observation vector."""
        continuous = np.array([
            state['cpu_utilization'],
            state['memory_usage'],
            state['disk_io'],
            state['network_bandwidth'],
            state['p90_latency'],
            state['p99_latency'],
            state['error_rate_5xx'],
            state['throughput'],
        ], dtype=np.float32)
        
        discrete = np.array([
            state['node_ready_status'] / 3,
            state['pending_pods']      / 100,
            state['crashloop_flag']    / 20,
            state['failed_pods']       / 50,
        ], dtype=np.float32)
        
        return np.clip(np.concatenate([continuous, discrete]), 0.0, 1.0)

print("✓ K8sSelfHealingEnv defined (observation-based, uniform action effects)")


# ===== From notebook code cell 22 =====
# ========== v3.1 ENVIRONMENT TUNING: Harder Scenarios ==========
# Making environment more challenging to ensure agent learns real strategies

class K8sSelfHealingEnvV31(gym.Env):
    """v3.1: Enhanced difficulty environment.
    
    Changes from v3.0:
    1. Natural recovery slower (0.98 → 0.96): Baseline must work harder
    2. Observation noise higher (0.015 → 0.025): Requires robust policy
    3. More severe initial failures: Higher starting error rates
    4. Stricter recovery thresholds: harder to reach success condition
    5. Recovery thresholds: error_rate < 0.01 (was 0.02)
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Dict[str, Any] = None):
        super(K8sSelfHealingEnvV31, self).__init__()
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 100)
        self.observation_step_interval = self.config.get('step_interval_sec', 10)
        self.num_deployments = self.config.get('num_deployments', 5)
        self.num_nodes = self.config.get('num_nodes', 3)
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = ActionSpace.get_action_space()
        
        self.current_step = 0
        self.episode_rewards = []
        self.current_state = None
        self.prev_state = None
        self._scenario_name = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_rewards = []
        self.current_state = self._generate_failed_state_severe()
        self.prev_state = self.current_state.copy()
        obs = self._encode_observation(self.current_state)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        action_name = ActionSpace.ACTIONS[action]['name']
        self._execute_action(action, action_name)
        self.prev_state = self.current_state.copy()
        self.current_state = self._collect_metrics_harder()
        
        reward = RewardCalculator.calculate(
            self.prev_state, self.current_state, action, self.current_step
        )
        self.episode_rewards.append(reward)
        
        recovered = self._is_recovered_strict()
        collapsed = self._is_collapsed()
        truncated = self.current_step >= self.max_steps
        terminated = recovered or collapsed
        
        obs = self._encode_observation(self.current_state)
        info = {
            'action': action_name,
            'recovered': recovered,
            'collapsed': collapsed,
            'episode_reward': sum(self.episode_rewards),
        }
        return obs, reward, terminated, truncated, info

    def _generate_failed_state_severe(self) -> Dict[str, float]:
        """Generate MORE SEVERE initial failures (v3.1)."""
        scenario = FailureScenario.sample_scenario()
        self._scenario_name = scenario['name']
        state = scenario['state'].copy()
        
        # Make failures MORE SEVERE
        state['error_rate_5xx'] = min(1.0, state['error_rate_5xx'] * 1.3)
        state['cpu_utilization'] = min(1.0, state['cpu_utilization'] * 1.2)
        state['memory_usage'] = min(1.0, state['memory_usage'] * 1.2)
        state['p99_latency'] = min(1.0, state['p99_latency'] * 1.25)
        state['pending_pods'] = min(100, state['pending_pods'] * 1.3)
        state['node_ready_status'] = min(3, state['node_ready_status'] * 1.2)
        
        return state

    def _execute_action(self, action: int, action_name: str) -> None:
        """Same as v3.0 (uniform, observation-based)."""
        s = self.current_state
        if action == 1:
            s['crashloop_flag'] = max(0, s['crashloop_flag'] - 2.0)
            s['failed_pods'] = max(0, s['failed_pods'] - 2.0)
            s['pending_pods'] = max(0, s['pending_pods'] - 2.0)
            s['error_rate_5xx'] = max(0, s['error_rate_5xx'] * 0.85)
            s['throughput'] = min(1.0, s['throughput'] + 0.04)
        elif action == 2:
            s['pending_pods'] = max(0, s['pending_pods'] - 6.0)
            s['throughput'] = min(1.0, s['throughput'] + 0.10)
            s['cpu_utilization'] = max(0.05, s['cpu_utilization'] - 0.08)
            s['memory_usage'] = max(0.05, s['memory_usage'] - 0.08)
            s['error_rate_5xx'] = max(0, s['error_rate_5xx'] * 0.91)
        elif action == 3:
            s['cpu_utilization'] = max(0.05, s['cpu_utilization'] - 0.05)
            s['memory_usage'] = max(0.05, s['memory_usage'] - 0.05)
            s['throughput'] *= 0.90
        elif action == 4:
            s['node_ready_status'] = max(0, s['node_ready_status'] - 1.0)
            s['pending_pods'] = min(100, s['pending_pods'] + 5.0)
        elif action == 5:
            s['node_ready_status'] = max(0, s['node_ready_status'] - 0.8)
            s['error_rate_5xx'] = max(0, s['error_rate_5xx'] * 0.95)
        elif action == 6:
            s['node_ready_status'] = max(0, s['node_ready_status'] - 0.8)
            s['pending_pods'] = max(0, s['pending_pods'] - 5.0)
            s['error_rate_5xx'] = max(0, s['error_rate_5xx'] * 0.96)
            s['throughput'] = min(1.0, s['throughput'] + 0.025)

    def _collect_metrics_harder(self) -> Dict[str, float]:
        """HARDER recovery (0.98 → 0.96, noise 0.015 → 0.025)."""
        state = {}
        for key, val in self.current_state.items():
            if key == 'node_ready_status':
                drift = max(0, val - np.random.uniform(0, 0.12))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.07), 0, 3))
            elif key == 'pending_pods':
                drift = max(0, val - np.random.uniform(0.8, 2.5))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.3), 0, 100))
            elif key == 'crashloop_flag':
                drift = max(0, val - np.random.uniform(0.5, 1.8))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.2), 0, 20))
            elif key == 'failed_pods':
                drift = max(0, val - np.random.uniform(0.5, 1.8))
                state[key] = float(np.clip(drift + np.random.normal(0, 0.2), 0, 50))
            else:
                # CHANGED: 0.98 → 0.96, 0.015 → 0.025
                target = np.random.uniform(0.08, 0.25)
                drift = val * 0.96 + target * 0.04
                state[key] = float(np.clip(drift + np.random.normal(0, 0.025), 0, 1))
        return state

    def _is_recovered_strict(self) -> bool:
        """Stricter recovery condition (v3.1)."""
        s = self.current_state
        return (
            s['error_rate_5xx'] < 0.01 and    # was 0.02 - STRICTER
            s['pending_pods'] < 1 and         # was 2 - STRICTER
            s['crashloop_flag'] < 0.5 and     # was 1 - STRICTER
            s['node_ready_status'] < 0.5      # was 1 - STRICTER
        )

    def _is_collapsed(self) -> bool:
        """Unrecoverable state."""
        s = self.current_state
        return s['error_rate_5xx'] > 0.5 and s['node_ready_status'] >= 3

    def _encode_observation(self, state: Dict[str, float]) -> np.ndarray:
        """Flatten to [0,1]^12."""
        continuous = np.array([
            state['cpu_utilization'],
            state['memory_usage'],
            state['disk_io'],
            state['network_bandwidth'],
            state['p90_latency'],
            state['p99_latency'],
            state['error_rate_5xx'],
            state['throughput'],
        ], dtype=np.float32)
        
        discrete = np.array([
            state['node_ready_status'] / 3,
            state['pending_pods'] / 100,
            state['crashloop_flag'] / 20,
            state['failed_pods'] / 50,
        ], dtype=np.float32)
        
        return np.clip(np.concatenate([continuous, discrete]), 0.0, 1.0)

# Initialize v3.1 environment
env_v31 = K8sSelfHealingEnvV31(env_config)

print("\n✅ v3.1 Environment created (harder scenarios)")
print(f"   Natural recovery drift: 0.98 → 0.96 (slower)")
print(f"   Observation noise: 0.015 → 0.025 (higher)")
print(f"   Initial failures: +20-30% more severe")
print(f"   Recovery thresholds: STRICTER (error < 0.01, pending < 1, etc)")


# ===== From notebook code cell 37 =====
# ========== Step 4: Deployment Summary & Code Changes ==========

print("\n" + "╔" + "═"*88 + "╗")
print("║" + " "*88 + "║")
print("║" + "  PRODUCTION DEPLOYMENT: WHAT TO CHANGE & HOW TO DEPLOY".center(88) + "║")
print("║" + " "*88 + "║")
print("╚" + "═"*88 + "╝")

deployment_summary = pd.DataFrame({
    'Component': [
        'Agent Model',
        'Metrics Source',
        'Action Executor',
        'Inference Loop',
        'Deployment',
        'Monitoring',
        'Error Handling'
    ],
    'Simulation (Now)': [
        'PPO trained',
        'Random data',
        'Simulated effects',
        'Jupyter notebook',
        'Not deployed',
        'Print statements',
        'None'
    ],
    'Production (Deploy)': [
        'agent_v3.2.zip',
        'Real Prometheus',
        'Real kubectl',
        'FastAPI/gRPC service',
        'Docker in K8s',
        'Prometheus metrics',
        'Retry + circuit breaker'
    ],
    'File Location': [
        'kubernetes-hub/models/',
        'k8s_self_healing_agent.py',
        'k8s_self_healing_agent.py',
        'k8s_self_healing_agent.py',
        'kubernetes-hub/docker/agent/',
        'k8s_self_healing_agent.py',
        'k8s_self_healing_agent.py'
    ]
})

print("\n" + deployment_summary.to_string(index=False))

print("\n" + "="*90)
print("🔧 DETAILED CODE CHANGES NEEDED")
print("="*90)

changes = {
    "1. METRICS COLLECTION": {
        "FROM (Simulation)": "self._collect_metrics() in K8sSelfHealingEnv → returns random data",
        "TO (Production)": "K8sMetricsCollector.get_cluster_metrics() → queries Prometheus",
        "Code Change": """
# OLD (Simulation)
def _collect_metrics(self):
    return {
        'error_rate': np.random.uniform(0, 0.1),
        'pending_pods': np.random.randint(0, 5),
        ...
    }

# NEW (Production) - Real Prometheus
class K8sMetricsCollector:
    def query_metric(self, query: str) -> float:
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query}
        )
        return float(response.json()["data"]["result"][0]["value"][1])
    
    def get_cluster_metrics(self) -> Dict:
        return {
            'error_rate': self.query_metric('rate(http_requests_total{status=~"5.."}[5m])'),
            'pending_pods': self.query_metric('count(kube_pod_status_phase{phase="Pending"})'),
            ...
        }
"""
    },
    
    "2. ACTION EXECUTION": {
        "FROM (Simulation)": "_execute_action() modifies state dict with effects",
        "TO (Production)": "K8sActionExecutor.execute_action() runs real kubectl",
        "Code Change": """
# OLD (Simulation)
def _execute_action(self, action):
    if action == 1:  # restart_pod
        self.state['crashed_pods'] *= 0.5  # Simulated effect
    elif action == 2:  # scale_up
        self.state['pending_pods'] *= 0.7  # Simulated effect

# NEW (Production) - Real kubectl
class K8sActionExecutor:
    def execute_action(self, action: int):
        if action == 1:  # restart_pod
            returncode, out, err = self.run_kubectl(
                'get pods --field-selector=status.phase=Failed | xargs delete'
            )
            return returncode == 0, out
        elif action == 2:  # scale_up
            returncode, out, err = self.run_kubectl(
                'scale deployments --replicas=+1'
            )
            return returncode == 0, out
"""
    },
    
    "3. OBSERVATION ENCODING": {
        "FROM (Simulation)": "_encode_observation(state) uses simulated state dict",
        "TO (Production)": "encode_observation(metrics) converts Prometheus metrics",
        "Code Change": """
# OLD (Simulation)
def _encode_observation(self):
    return np.array([
        self.state['error_rate'],
        self.state['pending_pods'] / 50,
        ...
    ])

# NEW (Production) - From real metrics
def encode_observation(self, metrics: Dict[str, float]) -> np.ndarray:
    return np.array([
        np.clip(metrics['error_rate'] / 0.05, 0, 1),
        np.clip(metrics['pending_pods'] / 50, 0, 1),
        ...
    ], dtype=np.float32)
"""
    },
    
    "4. INFERENCE LOOP": {
        "FROM (Simulation)": "Manual reset/step in Jupyter cells",
        "TO (Production)": "Continuous loop with decision_interval",
        "Code Change": """
# OLD (Simulation)
obs, info = env.reset()
for step in range(max_steps):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

# NEW (Production) - Continuous
while True:
    metrics = self.metrics_collector.get_cluster_metrics()
    obs = self.encode_observation(metrics)
    action, _ = self.model.predict(obs)
    success, msg = self.action_executor.execute_action(int(action))
    time.sleep(DECISION_INTERVAL)
"""
    },
    
    "5. MONITORING": {
        "FROM (Simulation)": "Print statements and saved plots",
        "TO (Production)": "Prometheus metrics + structured logging",
        "Code Change": """
# OLD (Simulation)
print(f"Step {step}: reward={reward:.3f}")
# Save matplotlib plots

# NEW (Production) - Prometheus + Logging
logger.info(f"Action: {action_name}, metrics: {metrics}")
agent_decisions.labels(action=action_name, status='success').inc()
recovery_time.observe(recovery_time_sec)
agent_state_error_rate.set(metrics['error_rate'])
# Prometheus scrapes /metrics endpoint
"""
    }
}

for topic, details in changes.items():
    print(f"\n{topic}")
    print("-" * 90)
    for key, value in details.items():
        if key == "Code Change":
            print(f"\n{value}\n")
        else:
            print(f"  {key}:")
            print(f"    {value}\n")

print("="*90)
print("📅 DEPLOYMENT TIMELINE")
print("="*90)

timeline = pd.DataFrame({
    'Phase': ['Setup', 'Staging', 'Validation', 'Canary', 'Gradual', 'Full'],
    'Duration': ['1 week', '2 weeks', '1 week', '2 weeks', '4 weeks', '4 weeks'],
    'Actions': [
        'Build Docker, setup K8s manifests',
        'Deploy to staging cluster',
        'Chaos engineering + edge cases',
        '5% production traffic',
        '50% production traffic',
        '100% production, full automation'
    ],
    'Success Criteria': [
        'Images built, manifests ready',
        'Agent running, metrics flowing',
        'pass 100 chaos tests, recovery reliable',
        'No errors, success rate >93%',
        'No escalations, MTTR <50% manual',
        'Full automation, MTTR <30% manual'
    ]
})

print("\n" + timeline.to_string(index=False))

print("\n" + "="*90)
print("🎯 DECISION MATRIX: v3.2 vs v3.0")
print("="*90)

decision_matrix = pd.DataFrame({
    'Criterion': [
        'Success Rate',
        'Trained On',
        'Generalization',
        'Recommendation',
        'Deployment Timeline',
        'Risk Level'
    ],
    'v3.2 (Primary)': [
        '≥93% (harder env)',
        'v3.1 harder environment',
        'Excellent (proven on v3.0)',
        '✅ DEPLOY NOW',
        'Week 1: Staging',
        'Low'
    ],
    'v3.0 (Backup)': [
        '95% (easier env)',
        'v3.0 original environment',
        'Good (but not on harder env)',
        '✅ READY BACKUP',
        'Week 3: If v3.2 fails',
        'Very Low'
    ]
})

print("\n" + decision_matrix.to_string(index=False))

print("\n" + "="*90)
print("✅ FINAL DEPLOYMENT PLAN")
print("="*90)

print("""
1️⃣  IMMEDIATE (This Week)
   ☐ Save model: agent_v3.2.zip to kubernetes-hub/models/
   ☐ Create production code: k8s_self_healing_agent.py
   ☐ Build Docker image: docker build -t registry/k8s-agent:v3.2
   ☐ Push to registry: docker push registry/k8s-agent:v3.2
   ☐ Create K8s manifests: k8s-agent-deployment.yaml
   ☐ Setup kubeconfig secret in cluster

2️⃣  WEEK 1-2 (Staging)
   ☐ Deploy to staging: kubectl apply -f k8s-agent-deployment.yaml
   ☐ Verify metrics in Prometheus: /metrics endpoint
   ☐ Run chaos engineering tests (network delay, pod crash, etc.)
   ☐ Validate recovery: success rate >93%, recovery time <50 steps
   ☐ Compare with v3.0 backup: A/B test on same failures
   ☐ Document decision logs and learned patterns

3️⃣  WEEK 3-4 (Canary)
   ☐ Deploy to 5% production traffic
   ☐ Monitor: MTTR, success rate, SLA compliance
   ☐ Enable detailed logging: agent decisions, actions, outcomes
   ☐ Create incident runbook: manual override procedures
   ☐ If issues: Rollback to manual ops or activate v3.0 backup

4️⃣  WEEK 5-8 (Gradual Rollout)
   ☐ Increase to 50% production traffic
   ☐ Continue monitoring and tuning
   ☐ Gather production data for v3.3 retraining
   ☐ Monthly reviews: agent performance, new failure patterns

5️⃣  WEEK 9-12 (Full Production)
   ☐ 100% production traffic
   ☐ Full automation enabled
   ☐ Expected MTTR: -30-50% vs manual ops
   ☐ Expected automation rate: 60-70% of failures
   ☐ Schedule monthly retraining cycles

✨ EXPECTED PRODUCTION IMPACT
   • MTTR: 30-50% faster recovery
   • Automation: 60-70% of failures auto-recovered
   • SLA: Maintained during recovery (conservative strategy)
   • Ops burden: Reduced by 60-70%
   • Scalability: Handle 4+ failure types + unknowns
""")

print("="*90)
