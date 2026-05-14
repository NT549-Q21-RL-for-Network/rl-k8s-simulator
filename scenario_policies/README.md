# scenario_policies

Thư mục này chứa code RL theo từng kịch bản lỗi. Quy ước hiện tại:

- File `.py` chỉ giữ logic chạy.
- Ghi chú thiết kế, giải thích MDP, và mapping file được đặt trong README.

## Cấu trúc

- `shared/core/`
  - `state_space.py`: định nghĩa state metrics, ranges, encode/decode.
  - `action_space.py`: định nghĩa tập action rời rạc cho agent.
  - `reward_calculator.py`: công thức reward và các hệ số.
  - `failure_scenarios.py`: mẫu trạng thái lỗi ban đầu.
  - `envs.py`: môi trường Gym nền (`K8sSelfHealingEnv`, `K8sSelfHealingEnvV31`).
- `shared/training/`
  - `baseline.py`: baseline policy để so sánh.
  - `evaluation.py`: evaluate agent theo nhiều episode.
  - `diagnostics.py`: hàm tiện ích phân tích kết quả train/eval.
- `shared/runtime/`
  - `k8s_runtime.py`: interface runtime để gắn Prometheus/K8s thật.
- `network_delay/`
  - `env_network_delay.py`: env ràng buộc lỗi `network_degradation`.
- `pod_failure/`
  - `env_pod_failure.py`: env ràng buộc lỗi `pod_crash_loop`.

## Nguyên tắc mở rộng

1. Logic dùng chung đặt trong `shared/*`.
2. Logic riêng theo fault đặt trong thư mục scenario tương ứng.
3. Không nhét ghi chú dài vào file `.py`; đưa vào README gần nhất.
