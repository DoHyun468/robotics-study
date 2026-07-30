# DreamerV3 (2023)

Hafner, Pasukonis, Ba, Lillicrap, *Mastering Diverse Domains through World Models*, 2023 (arXiv 2301.04104). [DreamerV2](dreamerv2.md)의 이산 잠재 RSSM + 상상(imagination) actor-critic 골격을 그대로 유지한 채, 견고성 기법을 총동원해 **하이퍼파라미터 하나로 여러 도메인**을 커버한다. 원형은 [Dreamer V1](dreamer.md).

## 한 줄 요약

성능 SOTA 자체보다 **견고성**(튜닝 없이 어떤 도메인이든 동일 하이퍼파라미터로 작동)에 초점을 맞춘 버전. symlog 예측·twohot 회귀·free bits·unimix·percentile return normalization 등 안정화 기법을 결합해 **8개 도메인 150+ 태스크**를 동일 설정으로 커버하고, **Minecraft 다이아몬드를 사람 데이터·커리큘럼 없이** 처음 채굴했다.

---

## 1. 문제

기본 문제 설정(POMDP·RSSM·상상 학습)은 [Dreamer V1](dreamer.md)·[DreamerV2](dreamerv2.md)와 동일하다 — 자세한 내용은 그쪽 참고.

DreamerV2까지도 도메인마다 **보상 스케일·관측 분포**가 달라 도메인별로 하이퍼파라미터를 다시 튜닝해야 했다. DreamerV3의 목표는 성능 SOTA 자체보다 **튜닝 없이 어떤 도메인이든** 동일 설정으로 도는 것 — 도메인마다 보상 스케일·관측 분포가 극단적으로 다른데도 **고정 하이퍼파라미터**로 돌게 만드는 것이 과제다.

---

## 2. 방법

World model은 [DreamerV2](dreamerv2.md)와 같은 **범주형 RSSM**을, actor-critic은 [Dreamer V1](dreamer.md)과 같은 상상 롤아웃 + λ-return 골격을 그대로 쓴다. Actor 그래디언트는 원문 명시대로 **연속 행동엔 stochastic backpropagation(pathwise), 이산 행동엔 REINFORCE**를 쓴다("stochastic backpropagation for continuous actions and reinforce for discrete actions"). 여기에 아래 견고성 기법들을 결합해 도메인 불문 하나의 하이퍼로 돌린다.

### 2.1 symlog 예측

보상·가치·벡터 관측처럼 스케일이 제각각인 타깃을 압착 함수로 정규화.

$$
\operatorname{symlog}(x) = \operatorname{sign}(x)\,\ln(1+|x|),\qquad \operatorname{symexp}(x)=\operatorname{sign}(x)\,(e^{|x|}-1)
$$

네트워크는 symlog 공간에서 예측하고 출력 시 symexp로 되돌린다. 큰 값의 gradient 폭발과 작은 값의 무시를 동시에 완화.

### 2.2 twohot 이산 회귀

스칼라(보상·가치)를 고정 bin 격자 위 **두 개의 이웃 bin에 선형 분배한 soft 분포**로 회귀 → 분류 손실로 학습. 회귀보다 넓은 동적 범위에 안정적.

$$
y \approx \sum_i p_i\, b_i,\qquad p\ \text{는 } y\text{를 감싸는 두 bin에만 질량을 두는 twohot 벡터}
$$

### 2.3 free bits + KL balancing

dynamics/representation 두 KL을 나누고([DreamerV2](dreamerv2.md)의 KL balancing 계승) 각각 **1 nat 아래로는 벌하지 않는다**(free bits). 작은 KL 과최적화로 인한 표현 붕괴 방지.

$$
\mathcal{L}_{\text{dyn}} = \max\!\big(1,\ \mathrm{KL}[\operatorname{sg}(q)\,\|\,p]\big),\qquad
\mathcal{L}_{\text{rep}} = \max\!\big(1,\ \mathrm{KL}[q\,\|\,\operatorname{sg}(p)]\big)
$$

전체 world model 손실(가중치 원문 값):

$$
\mathcal{L}(\phi) = \mathbb{E}\Big[\sum_t \beta_{\text{pred}}\mathcal{L}_{\text{pred}} + \beta_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \beta_{\text{rep}}\mathcal{L}_{\text{rep}}\Big],\quad \beta_{\text{pred}}{=}1,\ \beta_{\text{dyn}}{=}0.5,\ \beta_{\text{rep}}{=}0.1
$$

($\mathcal{L}_{\text{pred}}$ = 이미지/벡터 디코더 + 보상 + continue의 음의 로그가능도.)

### 2.4 unimix categoricals

범주형 확률에 **1% 균등분포를 섞어** 0 확률로 인한 KL 발산·죽은 클래스를 방지.

$$
p = 0.99\,\operatorname{softmax}(\text{logits}) + 0.01\,\tfrac{1}{K}
$$

### 2.5 percentile return normalization (actor)

상상 return을 배치의 **5~95 백분위 폭 $S$**로 나눠 스케일을 자동 정규화(단, $\max(1,S)$로 하한). 보상 크기가 도메인마다 달라도 actor 스텝 크기가 일정해진다.

$$
\tilde V^\lambda_\tau = \frac{V^\lambda_\tau}{\max\!\big(1,\ \operatorname{Per}_{95}(V^\lambda)-\operatorname{Per}_{5}(V^\lambda)\big)}
$$

actor는 $\tilde V^\lambda$를 최대화 + **고정 엔트로피 계수**로 탐험 유지.

### 2.6 스케일링

모델 크기를 키우면 성능·샘플효율이 **단조 상승**(작은 모델도 되지만 클수록 좋음) — 하나의 레시피로 크기만 키우면 됨을 보였다.

이 기법들의 순효과: **DMC(연속·이미지/상태)·Atari·ProcGen·Crafter·Minecraft** 등 8개 도메인 150+ 태스크를 **동일 하이퍼**로 커버하고, **Minecraft 다이아몬드를 사람 데이터·커리큘럼 없이** 처음 채굴했다.

---

## 3. 결과

수치는 원문(arXiv 2301.04104) 대조 완료.

**하나의 하이퍼파라미터 세트**로 8개 도메인 **150+ 태스크**에서 강력한 성능. 하이라이트: **Minecraft 다이아몬드를 from scratch**(사람 데이터·커리큘럼 없이) **최초 획득**. 모델 크기 스케일업이 성능을 단조 개선. (개별 벤치 절대 점수는 원문 부록 — 미기재.)

---

## 4. 한 줄 평·한계

**한 줄 평.** [Dreamer V1](dreamer.md)의 상상 학습, [DreamerV2](dreamerv2.md)의 이산 잠재를 그대로 물려받고, symlog·twohot·free bits·unimix·percentile normalization 같은 견고성 기법을 누적해 model-based RL을 "튜닝 지옥"에서 "레시피"로 바꿨다.

**한계.**
- **보상 신호 의존**: WM-RL은 보상이 정의된 태스크가 전제. 보상 설계가 어렵거나 없는 실세계 조작에는 그대로 안 옮겨진다(→ 모방/자기지도 표현형이 보완).
- **재구성 기반 표현**: V1/V2/V3 모두 관측 **재구성**으로 잠재를 학습 → 시각적으로 화려하지만 태스크에 무관한 픽셀 디테일에 표현 용량을 낭비할 수 있다. 이 지점을 **재구성 없이**(MuZero) 또는 **표현 예측만으로**(JEPA) 우회하는 갈래가 이후 흐름.
- **장기 상상의 오차 누적**: 상상 지평이 길어지면 prior 오차가 복리로 쌓인다. 근본 해결은 아니다.
- **실로봇 검증은 제한적**: DMC·Atari·ProcGen·Crafter·Minecraft 등 벤치(게임·시뮬 제어) 중심. 실제 매니퓰레이션·접촉 풍부 태스크로의 이식은 별도 과제(→ 로봇 실용 최전선은 예측형 V-JEPA-2 쪽).

---

## 부록: 버전 비교 요약

| 항목 | V1 (2020) | V2 (2021) | V3 (2023) |
|------|-----------|-----------|-----------|
| 주 도메인 | 연속제어(DMC 20태스크) | Atari 55게임 | 8도메인 150+ (Minecraft 포함) |
| 잠재 $z$ | 가우시안(연속) | 범주형 32×32 | 범주형 + unimix 1% |
| 잠재 gradient | reparameterization | straight-through | straight-through |
| KL 처리 | free nats(3) | KL balancing(α=0.8) | KL balancing + free bits(1 nat) |
| 보상/가치 회귀 | MSE(가우시안) | — | symlog + twohot |
| Actor gradient | pathwise(dynamics backprop) | 순수 REINFORCE(ρ=1, η=10⁻³) | 연속=pathwise, 이산=REINFORCE |
| Return 정규화 | 없음 | 없음 | 5–95 백분위, max(1,S) |
| continue 예측 | 없음(할인만) | 있음 | 있음 |
| 대표 성취 | 이미지 연속제어 SOTA(20태스크) | Atari 인간급, 단일 GPU(중앙값 2.15) | 하이퍼 하나로 150+ 태스크·Minecraft 다이아 |

> 위 표의 값은 원문(arXiv 1912.01603 / 2010.02193 / 2301.04104)에서 대조 완료. V3 하이퍼(β_pred=1, β_dyn=0.5, β_rep=0.1, free bits 1 nat, unimix 1%, 5–95 백분위)는 본문 기준. 원형은 [Dreamer V1](dreamer.md), 직전 버전은 [DreamerV2](dreamerv2.md) 참고.
