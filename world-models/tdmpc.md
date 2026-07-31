# TD-MPC (2022)

*재구성도 없고, 정책만도 아니다 — TD로 배운 잠재 모델 위에서 MPC*

Hansen, Wang, Su, *Temporal Difference Learning for Model Predictive Control* (TD-MPC), ICML 2022 (arXiv 2203.04955). 후속 TD-MPC2가 다중 태스크·대규모로 확장.

같은 갈래: [PlaNet](planet.md) · [Dreamer](dreamer.md) · [MuZero](muzero.md) · [Latent 개요](latent.md)

## 한 줄 요약

관측 재구성 없이 **보상·가치·잠재 일관성**만으로 잠재 동역학(TOLD)을 배우고, 그 위에서 **MPPI 계획 + 학습된 가치를 terminal로** 쓰는 하이브리드. "모델 기반 계획(단기 정밀) + TD 가치(장기 시야)"의 결합으로, 연속제어에서 SAC를 넘고 **DMControl Dog 태스크를 처음으로 푼** 방법이다 — [MuZero](muzero.md)의 재구성-프리 철학을 연속 행동 공간으로 가져온 실용판.

---

## 1. 문제 — 계획과 가치, 왜 하나만 골라야 하나

기존 구도의 양극단:

- **[PlaNet](planet.md)류 (모델+계획)**: 지평 $H$ 안은 정밀하지만 **그 너머를 못 본다**(근시안). 게다가 재구성 기반이라 태스크 무관 픽셀에 용량을 쓴다.
- **SAC류 (model-free 정책+가치)**: 장기는 가치함수가 커버하지만, 단기 행동 선택이 정책망 한 방이라 **테스트 시점의 추가 최적화 여지가 없다.**
- **[MuZero](muzero.md)**: 둘을 결합했지만 MCTS라 이산 행동 전용이고 무겁다.

> **TD-MPC의 답**: 재구성 없는 잠재 모델을 배우되, 지평 안은 **MPPI로 계획**하고 지평 끝은 **TD로 배운 $Q$가 이어받는다.** 단기 정밀 + 장기 시야를 연속 행동에서 동시에.

## 2. 방법

### 2.1 TOLD — Task-Oriented Latent Dynamics

다섯 부품(전부 잠재공간, 디코더 없음):

$$
\begin{aligned}
&\text{Encoder:} && z_t = h_\theta(s_t)\\
&\text{Latent dynamics:} && \hat z_{t+1} = d_\theta(z_t, a_t)\\
&\text{Reward:} && \hat r_t = R_\theta(z_t, a_t)\\
&\text{Value:} && \hat q_t = Q_\theta(z_t, a_t)\\
&\text{Policy (prior):} && \hat a_t \sim \pi_\theta(z_t)
\end{aligned}
$$

**손실 세 개를 공동 최적화**(다스텝 언롤에 시간 가중, BPTT):

$$
\underbrace{\|\hat r - r\|^2}_{\text{보상}} \;+\; c_2\underbrace{\big\|Q_\theta(z,a) - \big(r + \gamma Q_{\theta^-}(z',\pi_\theta(z'))\big)\big\|^2}_{\text{가치 TD}} \;+\; c_3\underbrace{\big\|d_\theta(z,a) - h_{\theta^-}(s')\big\|^2}_{\text{잠재 일관성}}
$$

셋째 항이 이 논문의 재료다 — "**다음 잠재 예측이, 다음 관측을 인코딩한 것과 같아야 한다**"는 잠재 일관성(latent consistency). 재구성 없이도 동역학이 관측에 정박(anchor)되는 장치로, [MuZero](muzero.md)(정박 없음, 보상·가치로만 간접 형성)와 [PlaNet](planet.md)(픽셀 재구성으로 정박)의 정확히 중간 지점이다.

### 2.2 계획 — MPPI + 가치 terminal + 정책 warm-start

매 스텝, 잠재공간에서 행동 시퀀스를 샘플 기반 최적화(MPPI):

$$
\phi_\Gamma = \mathbb{E}\Big[\ \underbrace{\textstyle\sum_{t}^{H} \gamma^t R_\theta(z_t,a_t)}_{\text{지평 안: 모델로 정밀 평가}} \;+\; \underbrace{\gamma^H Q_\theta(z_H, a_H)}_{\text{지평 밖: 가치가 이어받음}}\ \Big]
$$

- **지평이 짧아도 된다**: $H=5$(Dog/Humanoid는 반복만 8~12회) — terminal $Q$가 나머지 미래를 요약하므로 [PlaNet](planet.md)의 $H=12$ 근시안 문제가 사라진다.
- **정책 $\pi_\theta$는 계획의 씨앗**: $\pi$로 뽑은 후보 궤적을 샘플 풀에 섞어 warm-start — 정책은 "답"이 아니라 "좋은 초기값 공급자"다.
- 탐험은 샘플 분산 하한 $\max(\sigma, \epsilon)$, $\epsilon$을 0.5→0.05로 선형 감쇠.

## 3. 결과 (원문 대조)

| 항목 | 결과 |
|------|------|
| DMControl 15태스크 (상태 입력) | SAC·LOOP 대비 일관 우위(특히 Quadruped·Acrobot 같은 고난도) |
| **Dog 태스크** | **최초 해결 보고** — Humanoid/Dog급 고차원 보행을 ~1M 스텝에 |
| 이미지 입력 6태스크 | DrQ·DreamerV2와 동급(예: Walker Walk 100k에서 577±208 vs DrQ 612±164) — 태스크별 튜닝 없이 |
| 규모 | 파라미터 ~1.5M(Walker 기준) — LOOP 대비 wall-time 16배 빠름 |

작은 모델(1.5M)로 고차원 연속제어를 푼다는 점이 실용 포인트 — 대신 매 스텝 계획을 도는 비용은 남는다(Humanoid Stand 500k 스텝에 9.39h vs SAC 1.82h).

## 4. 스터디와의 개념적 연결

- **세 갈래의 종합**으로 읽는 게 정석이다: 재구성-프리는 [MuZero](muzero.md)에서, 잠재공간 MPC는 [PlaNet](planet.md)에서, 정책·가치 상각은 [Dreamer](dreamer.md)에서 — 각각의 강점만 연속제어용으로 재조립했다. Latent 갈래를 다 읽은 뒤 "그래서 로봇 연속제어엔 뭘 쓰나"에 대한 2022년의 답.
- **"계획 + terminal 가치" 구조**는 이 사이트의 파이프라인 관점과 잘 붙는다 — 우리가 [manipulation](../manipulation.md)에서 IK로 "남은 거리"를 한 번에 풀 듯, TD-MPC는 지평 밖 미래를 $Q$ 한 번으로 요약한다. 지평 안팎의 분업이라는 설계 감각이 같다.
- 실제 로봇 RL에서 TD-MPC(2) 계열은 Dreamer류와 함께 표준 베이스라인로 자리잡았다 — 보상 기반 축의 현재형이라는 위치.

## 5. 한 줄 평·한계

**한 줄 평.** "지평 안은 모델이, 지평 밖은 가치가" — 재구성 없는 잠재 모델과 TD 가치를 MPPI로 묶어, 연속제어에서 model-based의 정밀함과 model-free의 시야를 동시에 얻은 실용적 종합. Dog 최초 해결이라는 결과가 설계의 정당성을 증명한다.

**한계.**
- **추론 비용**: 매 스텝 MPPI 반복 — SAC 대비 wall-time 수 배(Humanoid 9.39h vs 1.82h). 고주파 실로봇 루프엔 부담.
- **보상 의존**: 잠재가 보상·가치로만 형성되므로 보상이 성긴/없는 태스크에선 표현이 빈약해질 수 있다(재구성의 반대 극단이 갖는 위험).
- **탐험**: MPPI 분산 하한 + TD의 조합은 원리적 탐험이 아니다 — 희소 보상 탐험 문제는 미해결.
- **단일 태스크 프레임**: 원판은 태스크당 학습 — 다중 태스크·스케일업은 TD-MPC2의 몫(이 리뷰 범위 밖).
