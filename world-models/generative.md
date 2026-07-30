# Generative World Models

관측(영상)을 **직접 생성**하는 인터랙티브 world model. 스케일·화제성은 최고이나 **장기 물리 일관성·정밀 제어는 아직** — 로봇 제어기보다 sim/데이터 생성 용도가 먼저. 개념적 위치는 [Concepts](../concepts.md).

- [GameNGen (2024)](gamengen.md) — 신경망만으로 DOOM 실시간 시뮬레이션.
- [Genie (2024)](genie.md) — 라벨 없는 영상에서 잠재 행동을 배우는 조작 가능한 생성 환경.
- [DIAMOND (2024)](diamond.md) — diffusion world model 안에서 배우는 Atari100k 에이전트.
- [Cosmos (2025)](cosmos.md) — Physical AI를 위한 대규모 world foundation model(NVIDIA).

## 비교표 — 같은 "생성"이지만 목적이 네 갈래

| 모델 | 연도 | 생성 방식 | 무엇을 위해 생성하나 | 대표 수치 (원문 대조) |
|------|------|-----------|---------------------|----------------------|
| [DIAMOND](diamond.md) | 2024 | EDM 확산 (1~3 step) | **RL 에이전트의 상상 학습장** | Atari 100k mean HNS 1.46 (world-model-only 신기록, 11/26게임 초인간) |
| [GameNGen](gamengen.md) | 2024 | SD 1.4 개조 확산 (4-step DDIM) | **게임 엔진 대체** (사람이 플레이) | DOOM 20 FPS(TPU-v5 1장), PSNR 29.43, 사람 구분율 58~60% |
| [Genie](genie.md) | 2024 | 자기회귀 토큰 (10.7B) | **행동 라벨 없이 제어 가능한 세계** | 잠재 행동 \|A\|=8, Platformers 3만 시간(6.8M 영상)으로 학습 |
| [Cosmos](cosmos.md) | 2025 | 확산(7B·14B) + AR(4B~13B) | **Physical AI용 데이터·시뮬 인프라** | 2,000만 시간 → 10⁸ 클립 사전학습, open-weight(CC BY 4.0) |

**로봇 관점 요약**: 제어기로 바로 쓸 수 있는 건 아직 없다. DIAMOND는 "생성 화질이 정책 성능이 된다"는 근거를, GameNGen·Genie는 상호작용 세계 생성의 상한을, Cosmos는 실용 경로(데이터 공급)를 각각 담당한다. 정밀 제어가 필요하면 [Predictive](predictive.md)(V-JEPA 2-AC)가, 보상 기반 학습이 필요하면 [Latent](latent.md)(Dreamer류)가 현재로선 더 가깝다.

```{note}
**Sequence-Based World Models**(Decision Transformer · Trajectory Transformer)는 [별도 카테고리](sequence.md)에서 다룬다.
```
