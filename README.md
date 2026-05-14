# rl-k8s-simulator

Mô phỏng RL offline cho bài toán self-healing Kubernetes, tổ chức theo fault scenario để dễ train/eval/deploy tách biệt.

## Cấu trúc repo

- `rl-training.ipynb`: notebook gốc.
- `scenario_policies/`: mã Python đã chuẩn hóa theo kịch bản lỗi.
- `training_results/`: kết quả huấn luyện, metadata.
- `training_summary.json`: summary run gần nhất.

Chi tiết module ở `scenario_policies/README.md`.

## Quy ước code

- File `.py` chỉ chứa logic thực thi.
- Ghi chú thiết kế đặt trong README (repo hoặc thư mục con).
- Khi cần giải thích hành vi một module, README phải chỉ rõ file tương ứng.

## MDP hiện tại (theo code đang dùng)

### 1) State space `S`

Observation là vector 12 chiều, chuẩn hoá về `[0, 1]`:

- Continuous (8):
  - `cpu_utilization`, `memory_usage`, `disk_io`, `network_bandwidth`
  - `p90_latency`, `p99_latency`, `error_rate_5xx`, `throughput`
- Discrete-like (4):
  - `node_ready_status` (`[0, 3]`)
  - `pending_pods` (`[0, 100]`)
  - `crashloop_flag` (`[0, 20]`)
  - `failed_pods` (`[0, 50]`)

File liên quan:
- `scenario_policies/shared/core/state_space.py`
- `scenario_policies/shared/core/envs.py`

### 2) Action space `A`

Action rời rạc gồm 7 hành động:
- `idle`, `restart_pod`, `scale_up`, `scale_down`, `drain_node`, `cordon_node`, `uncordon_node`

File liên quan:
- `scenario_policies/shared/core/action_space.py`

### 3) Transition `P(s'|s,a)`

State mới hình thành từ:
- tác động của action,
- drift hồi phục tự nhiên,
- nhiễu ngẫu nhiên.

Scenario khởi tạo episode (base):
- `node_failure`, `pod_crash_loop`, `resource_exhaustion`, `network_degradation`

Scenario-specific env:
- `network_delay/env_network_delay.py`: chỉ lấy `network_degradation`
- `pod_failure/env_pod_failure.py`: chỉ lấy `pod_crash_loop`

File liên quan:
- `scenario_policies/shared/core/failure_scenarios.py`
- `scenario_policies/shared/core/envs.py`
- `scenario_policies/network_delay/env_network_delay.py`
- `scenario_policies/pod_failure/env_pod_failure.py`

### 4) Reward `R(s,a,s')`

Công thức:
`R = alpha * stability_gain - beta * overhead_cost - gamma * action_disruption - delta * sla_penalty - step_penalty`

Hệ số hiện tại:
- `alpha = 8.0`
- `beta = 1.5`
- `gamma = 2.0`
- `delta = 4.0`
- `step_penalty = 0.06`

File liên quan:
- `scenario_policies/shared/core/reward_calculator.py`

### 5) Episode termination

Episode kết thúc khi:
- recovered, hoặc
- collapsed, hoặc
- chạm `max_steps`.

Mặc định thường dùng:
- `max_steps = 100`
- `step_interval_sec = 10`

File liên quan:
- `scenario_policies/shared/core/envs.py`

## Huấn luyện hiện tại

- Thuật toán chính: PPO
- Hyperparameters điển hình:
  - `total_timesteps = 100000`
  - `learning_rate = 3e-4`
  - `n_steps = 2048`
  - `batch_size = 64`
  - `gamma = 0.99`
  - `gae_lambda = 0.95`

Theo dõi kết quả tại:
- `training_summary.json`
- `training_results/run_*/training_metadata.json`

## Chạy train nhanh

1. Tạo môi trường:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train scenario `pod_failure`:

```bash
python scripts/train_ppo.py --scenario pod_failure --timesteps 30000 --eval-episodes 20
```

3. Train scenario `network_delay`:

```bash
python scripts/train_ppo.py --scenario network_delay --timesteps 30000 --eval-episodes 20
```

Kết quả sẽ nằm ở:
- `training_results/run_<timestamp>_<scenario>/ppo_model.zip`
- `training_results/run_<timestamp>_<scenario>/training_metadata.json`
- `training_summary.json`

## CI/CD build và push Docker image

Workflow: `.github/workflows/docker-build-push.yml`

- Trigger: mỗi lần `push` commit lên bất kỳ branch nào.
- Registry: `ghcr.io/${OWNER}/${REPO}`.
- Tags sinh tự động:
  - `sha-<commit_sha>`
  - `<branch-name>`
  - `latest` (chỉ branch mặc định).

Lệnh pull image sau khi workflow chạy xong:

```bash
docker pull ghcr.io/nt549-q21-rl-for-network/rl-k8s-simulator:latest
```
