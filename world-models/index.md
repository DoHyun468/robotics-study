# World Models — 심화 리뷰

Dreamer, PlaNet, MuZero, V-JEPA, 그리고 생성·영상형 world model(DIAMOND·GameNGen·Genie·Cosmos 등)까지 — world model 계열을 다루는 심화 리뷰 모음이다. [Paper Reviews](../reviews/index.md)가 직접 돌리거나 실습한 논문을 다루는 것과 달리, 이 트랙은 world model의 최전선을 개념적으로 정리한다. 수치·주장은 논문 원문 기준으로만 적고, 확인되지 않은 값은 "미확인"으로 표기한다.

## World Model 3축

| 축 | 대표 | 성숙도 | 로봇 관점 |
|----|------|--------|-----------|
| 예측·표현형 | JEPA, V-JEPA(-2) | 성숙, 최전선 | 실제 로봇 계획에 사용 — 3D/기하 강점과 가장 가까움 |
| 시퀀스·토큰형 | Dreamer, MuZero, TD-MPC | 성숙, 샘플효율↑ | 보상 기반 model-based RL |
| 생성·영상형 | DIAMOND, GameNGen, Genie, Cosmos | 화려하나 제어 미흡 | sim/데이터 생성 용도가 먼저 |

**VLA(모방)** vs **WM-RL(보상)**: 다른 문제·다른 신호라 같은 벤치로 직접 비교 불가 — 자세한 비교는 [Concepts](../concepts.md) 참고.

## 목록

1. [Dreamer V1/V2/V3](dreamer.md) — RSSM latent WM + 상상 rollout에서 actor-critic.
2. [PlaNet](planet.md) — Dreamer 전신. RSSM 동역학만 + CEM 온라인 계획(정책망 없음).
3. [MuZero](muzero.md) — 관측 재구성 없이 보상·가치·정책만 예측 + MCTS.
4. [V-JEPA / V-JEPA-2](vjepa.md) — 예측·표현형, 실제 로봇 계획.
5. [생성·영상형 WM](generative-wm.md) — DIAMOND / GameNGen / Oasis / Cosmos / Genie(-3).

## 출처

핵심 수치는 arXiv 원문에서 직접 대조해 채웠다. 대조 출처:
Dreamer V1 [1912.01603] / V2 [2010.02193] / V3 [2301.04104] / PlaNet [1811.04551] / MuZero [1911.08265] / V-JEPA 2 [2506.09985] / DIAMOND [2405.12399] / GameNGen [2408.14837] / Genie [2402.15391] / Cosmos [2501.03575].

남은 미확인은 **Genie-3 세부**(기술 리포트 미공개)뿐 — 그 외는 원문 대조 완료.
