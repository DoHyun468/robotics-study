# Latent World Models

관측을 잠재상태로 압축해 **잠재 동역학**(다음 잠재상태·보상)을 배우고, 그 안에서 계획하거나 상상 궤적으로 정책을 학습한다. 보상 기반 model-based RL의 중심 계열. 개념적 위치는 [Concepts](../concepts.md).

- [PlaNet (2019)](planet.md) — RSSM 동역학만 배우고 CEM으로 온라인 계획(정책망 없음). Dreamer의 전신.
- [Dreamer (V1, 2020)](dreamer.md) — RSSM + **상상 rollout**에서 actor-critic. 연속제어(DMC).
- [MuZero (2020)](muzero.md) — 관측 재구성 없이 보상·가치·정책만 예측 + **MCTS**.
- [DreamerV2 (2021)](dreamerv2.md) — 이산 잠재 world model, Atari 인간급.
- [DreamerV3 (2023)](dreamerv3.md) — 하이퍼파라미터 하나로 150+ 태스크, Minecraft 다이아몬드 from scratch.
