# Latent World Models

관측을 잠재상태로 압축해 **잠재 동역학**(다음 잠재상태·보상)을 배우고, 그 안에서 계획하거나 상상 궤적으로 정책을 학습한다. 보상 기반 model-based RL의 중심 계열. 개념적 위치는 [Concepts](../concepts.md).

## 계보

```
World Models (2018) ──→ PlaNet (2019) ──→ Dreamer (2020) ──→ DreamerV2 (2021) ──→ DreamerV3 (2023)
 V·M·C, 꿈속 훈련       RSSM+CEM 계획      +상상 actor-critic    +이산 잠재(Atari)     +견고성(150+ 태스크)

MuZero (2020)  ← 재구성 없이 보상·가치·정책만 예측 + MCTS
TD-MPC (2022)  ← 재구성-프리 + MPPI 계획 + TD 가치 terminal (연속제어 하이브리드)
```

- [World Models (2018)](world-models-2018.md) — VAE+MDN-RNN+867개 파라미터 컨트롤러, **꿈속 훈련**의 원조. 분야의 이름이 된 논문.
- [PlaNet (2019)](planet.md) — RSSM 동역학만 배우고 CEM으로 온라인 계획(정책망 없음). Dreamer의 전신.
- [Dreamer (V1, 2020)](dreamer.md) — RSSM + **상상 rollout**에서 actor-critic. 연속제어(DMC).
- [MuZero (2020)](muzero.md) — 관측 재구성 없이 보상·가치·정책만 예측 + **MCTS**.
- [DreamerV2 (2021)](dreamerv2.md) — 이산 잠재 world model, Atari 인간급.
- [TD-MPC (2022)](tdmpc.md) — 재구성-프리 잠재 모델 + **MPPI 계획 + TD 가치 terminal**. DMControl Dog 최초 해결.
- [DreamerV3 (2023)](dreamerv3.md) — 하이퍼파라미터 하나로 150+ 태스크, Minecraft 다이아몬드 from scratch.

## 비교표

| 모델 | 연도 | 핵심 기여 | 행동 결정 | 대표 수치 (원문 대조) |
|------|------|-----------|-----------|----------------------|
| [World Models](world-models-2018.md) | 2018 | V·M·C 분해, 꿈속 훈련·모델 착취 문제 제기 | CMA-ES 진화 (867 파라미터) | CarRacing 906±21 최초 해결, 꿈→실환경 이식(VizDoom 1092) |
| [PlaNet](planet.md) | 2019 | RSSM(결정적+확률적 잠재) 제안, 잠재공간 계획 | CEM 온라인 계획 (H=12, J=1000) | DMC 6태스크, 100 에피소드로 A3C의 10만 에피소드 능가 |
| [Dreamer](dreamer.md) | 2020 | 상상 궤적 위 actor-critic, pathwise gradient | 학습된 정책 (상각) | DMC 20태스크, PlaNet·D4PG 대비 효율·성능 우위 |
| [MuZero](muzero.md) | 2020 | 재구성 없는 모델(보상·가치·정책만) + MCTS | MCTS (보드 800/Atari 50 sims) | Atari 57게임 median 2041%(vs R2D2 1921%), 바둑·체스·쇼기 AlphaZero급 |
| [DreamerV2](dreamerv2.md) | 2021 | 범주형 잠재 32×32, KL balancing(α=0.8) | 학습된 정책 (REINFORCE) | Atari 55게임 median 2.15 — 단일 GPU로 IQN·Rainbow 능가 |
| [TD-MPC](tdmpc.md) | 2022 | 잠재 일관성 손실 + 계획·가치 하이브리드 | MPPI (H=5) + 가치 terminal | DMControl Dog 최초 해결, SAC·LOOP 능가 (~1.5M 파라미터) |
| [DreamerV3](dreamerv3.md) | 2023 | symlog·twohot·free bits — 튜닝 없는 견고성 | 학습된 정책 (연속 pathwise/이산 REINFORCE) | 고정 하이퍼로 150+ 태스크, Minecraft 다이아 from scratch 최초 |

**읽는 순서 제안**: World Models(원조와 문제 틀) → PlaNet(RSSM이 왜 생겼나) → Dreamer(상상 학습의 원형) → V2·V3(견고성 기법 누적) → MuZero(재구성을 버린 반대편 답) → TD-MPC(계획과 가치의 종합, 연속제어의 현재형). 재구성의 한계가 궁금해지면 [Predictive](predictive.md)(표현만 예측)와 [Generative](generative.md)의 [DIAMOND](diamond.md)(반대로 재구성을 더 잘하기)로, "모델·정책·가치를 아예 하나의 시퀀스 모델로" 쪽이 궁금해지면 [Sequence-Based](sequence.md)(DT·TT·IRIS)로 이어진다.
