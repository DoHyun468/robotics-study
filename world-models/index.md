# World Models — 심화 리뷰

world model 계열을 **유형별로** 정리한 심화 리뷰 트랙이다. [Paper Reviews](../reviews/index.md)가 직접 돌리거나 실습한 논문을 다루는 것과 달리, 이 트랙은 world model의 최전선을 개념적으로 정리한다. 수치·주장은 논문 원문(arXiv) 기준으로만 적고, 확인되지 않은 값은 "미확인"으로 표기한다. 개념적 위치는 [Concepts](../concepts.md) 참고.

## 유형 분류

world model은 "무엇을 어떻게 예측/생성하는가"에 따라 네 갈래로 나뉜다.

| 유형 | 무엇을 배우나 | 대표 | 로봇 관점 |
|------|--------------|------|-----------|
| **Latent** | 관측을 잠재상태로 압축해 **잠재 동역학**을 학습, 그 안에서 계획/상상 | PlaNet · Dreamer · MuZero | 보상 기반 model-based RL, 샘플효율 |
| **Sequence-Based** | 궤적(상태·행동·보상)을 **시퀀스 모델링** | Decision/Trajectory Transformer | offline RL을 시퀀스 예측으로 |
| **Predictive** | 픽셀 재구성 없이 **표현공간에서 미래 예측**(JEPA) | V-JEPA · V-JEPA 2 | 실제 로봇 계획 — 3D/기하 강점과 가장 가까움 |
| **Generative** | 관측(영상)을 **직접 생성**하는 인터랙티브 시뮬레이터 | GameNGen · Genie · DIAMOND · Cosmos | 화려하나 제어 미흡, sim/데이터 생성이 먼저 |

**VLA(모방)** vs **WM-RL(보상)**: 다른 문제·다른 신호라 같은 벤치로 직접 비교 불가 — [Concepts](../concepts.md) 참고.

## Latent World Models

잠재 동역학을 배워 그 안에서 계획하거나 상상 궤적으로 정책을 학습한다.

- [PlaNet (2019)](planet.md) — RSSM 동역학만 배우고 CEM으로 온라인 계획(정책망 없음). Dreamer의 전신.
- [Dreamer (V1, 2020)](dreamer.md) — RSSM + **상상 rollout**에서 actor-critic. 연속제어(DMC).
- [MuZero (2020)](muzero.md) — 관측 재구성 없이 보상·가치·정책만 예측 + **MCTS**.
- [DreamerV2 (2021)](dreamerv2.md) — 이산 잠재 world model, Atari 인간급.
- [DreamerV3 (2023)](dreamerv3.md) — 하이퍼파라미터 하나로 150+ 태스크, Minecraft 다이아몬드.

## Predictive World Models

픽셀을 재구성하지 않고 표현공간에서 미래를 예측한다(JEPA 계열).

- [V-JEPA (2024)](vjepa.md) — 비디오 자기지도 예측 표현학습.
- [V-JEPA 2 (2025)](vjepa2.md) — 대규모 확장 + **실제 로봇 계획**(V-JEPA 2-AC, Franka zero-shot).

## Generative World Models

관측(영상)을 직접 생성하는 인터랙티브 world model.

- [GameNGen (2024)](gamengen.md) — 신경망만으로 DOOM 실시간 시뮬레이션.
- [Genie (2024)](genie.md) — 잠재 행동으로 조작 가능한 생성 환경.
- [DIAMOND (2024)](diamond.md) — diffusion world model로 Atari100k.
- [Cosmos (2025)](cosmos.md) — 대규모 물리 world model 파운데이션.

```{note}
**Sequence-Based World Models**(Decision Transformer · Trajectory Transformer)는 아직 이 트랙에서 다루지 않았다 — 추후 추가 예정.
```

## 출처

핵심 수치는 arXiv 원문에서 직접 대조했다: Dreamer V1 [1912.01603] · V2 [2010.02193] · V3 [2301.04104] · PlaNet [1811.04551] · MuZero [1911.08265] · V-JEPA [2404.08471] · V-JEPA 2 [2506.09985] · DIAMOND [2405.12399] · GameNGen [2408.14837] · Genie [2402.15391] · Cosmos [2501.03575]. 남은 미확인은 **Genie-3 세부**(기술 리포트 미공개)뿐.
