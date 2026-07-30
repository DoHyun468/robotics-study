# DreamerV2 (2021)

Hafner, Lillicrap, Norouzi, Ba, *Mastering Atari with Discrete World Models*, ICLR 2021 (arXiv 2010.02193). [Dreamer V1](dreamer.md)의 RSSM + 상상(imagination) actor-critic 골격 위에 이산(categorical) 잠재와 KL balancing을 얹어, Atari에서 인간급을 처음 달성했다. 이후 [DreamerV3](dreamerv3.md)가 도메인 불문 견고성으로 더 확장한다.

## 한 줄 요약

World model을 잠재공간에서 학습하고 **행동은 전부 잠재 상상에서** 배운 최초의 에이전트로, Atari 55게임·200M 프레임에서 인간급 성능을 달성했다. V1의 가우시안 잠재·pathwise 그래디언트를 **이산(categorical) 잠재 + KL balancing + REINFORCE actor**로 바꾼 것이 핵심.

---

## 1. 문제

기본 문제 설정(POMDP·"왜 상상에서 학습이 이득인가")은 [Dreamer V1](dreamer.md)과 동일하다 — 자세한 내용은 그쪽 1절 참고.

DreamerV2가 겨냥한 구체적 간극: V1의 **가우시안 잠재 + pathwise 그래디언트** 조합은 연속제어엔 잘 맞았지만, Atari처럼 **다봉(multimodal)·불연속적인 동역학**(장면 전환, 점수 점프)을 갖고 **행동 자체가 이산**인 도메인에는 맞지 않는다. 이산 잠재와, 이산 행동에 맞는 학습 신호(REINFORCE)가 필요했다.

---

## 2. 방법

World model은 V1과 동일한 **RSSM** 골격(결정적 $h_t$ + 확률적 $z_t$, prior/posterior 분리)을 유지하고 상상 롤아웃 위에서 actor-critic을 학습한다(전체 알고리즘 루프는 [Dreamer V1](dreamer.md) 2.4절과 동일). 바뀐 핵심은 다음 3가지.

### 2.1 이산(categorical) 잠재

$z_t$를 가우시안 대신 **32개 범주형 변수 × 각 32 클래스**의 벡터로 둔다(one-hot 32×32). 샘플링은 미분 불가라 **straight-through gradient**를 쓴다: forward는 one-hot 샘플, backward는 확률(softmax)로 흘린다.

$$
z = \operatorname{onehot}(\arg\max)\quad\text{(forward)},\qquad \nabla \leftarrow \nabla\,\text{softmax logits}\quad\text{(straight-through backward)}
$$

이산 잠재가 Atari 같은 다봉·불연속 동역학(장면 전환, 점수 점프)에 가우시안보다 잘 맞았다는 게 경험적 발견이다.

### 2.2 KL balancing

KL 항을 prior/posterior 두 방향으로 나눠 **서로 다른 학습률**을 준다. prior를 posterior로 끌어당기는 쪽(동역학 학습)을 더 세게, 반대는 약하게.

$$
\mathcal{L}_{\mathrm{KL}} = \alpha\,\mathrm{KL}\!\big[\operatorname{sg}(q)\,\|\,p\big] + (1-\alpha)\,\mathrm{KL}\!\big[q\,\|\,\operatorname{sg}(p)\big],\qquad \alpha=0.8
$$

앞 항(prior가 움직임)에 가중치를 크게. representation이 prior를 맞추려 정보를 버리는 붕괴를 막는다. (V1의 free nats를 대체하는 장치 — [Dreamer V1](dreamer.md) 2.2절 참고.)

### 2.3 이산 행동 Actor = REINFORCE + entropy

연속 행동에 쓰던 pathwise(dynamics backprop) 그래디언트는 이산 행동엔 안 통한다. 대신 **REINFORCE**(score-function) + 엔트로피 정규화를 쓴다.

$$
\nabla_\theta \approx \mathbb{E}\Big[\sum_\tau \ln\pi_\theta(a_\tau)\,\operatorname{sg}(V^\lambda_\tau - v_\psi) \;+\; \eta\,\nabla_\theta\mathrm{H}[\pi_\theta(\cdot\mid h_\tau,z_\tau)]\Big]
$$

($V^\lambda$는 [Dreamer V1](dreamer.md) 2.3절과 동일한 λ-return.) 그 외 **continue predictor**($\hat c_t \sim p_\phi(\hat c_t \mid h_t, z_t)$, $c_t\in\{0,1\}$)를 새로 도입해 에피소드 종료를 상상 안에서 명시적으로 모델링한다(V1은 discount factor로만 처리).

결과적으로 **단일 GPU, Atari 200M 프레임 예산**에서 Rainbow·IQN 등 강한 model-free를 같은 예산 내에서 능가한다.

---

## 3. 결과

수치는 원문(arXiv 2010.02193) 대조 완료. world model을 픽셀 재구성으로 학습하고 **행동은 전부 잠재 상상에서** 배운 최초의 에이전트다.

| 항목 | 값 (원문) |
|------|-----------|
| 벤치 | Atari **55게임 · 200M 프레임 · 단일 GPU** |
| gamer-normalized 중앙값 | **2.15** (인간=1.0) |
| gamer-normalized 평균 | 11.33 |
| clipped record mean (원문 권장 지표) | 0.28 |
| 비교 | 동일 연산예산·wall-clock에서 top 단일-GPU 에이전트 **IQN·Rainbow 능가** |
| actor 설정 | 순수 REINFORCE(ρ=1), 엔트로피 η=10⁻³, KL balancing α=0.8 |

"상상 속에서만 배웠는데 model-free 최강을 이긴다"가 헤드라인 — V1이 연속제어에서 연 길이 이산·다봉 도메인에서도 성립함을 보인 수치다.

---

## 4. 한 줄 평·한계

**한 줄 평.** 이산 잠재 + KL balancing + REINFORCE로, [Dreamer V1](dreamer.md)의 상상 학습 골격을 처음 Atari급 이산·다봉 동역학에 확장했다 — 단일 GPU로 강한 model-free 베이스라인을 넘어선 첫 사례. 견고성은 이후 [DreamerV3](dreamerv3.md)에서 도메인 불문으로 더 확장된다.

**한계.**
- **보상 신호 의존**: WM-RL은 보상이 정의된 태스크가 전제. 보상 설계가 어렵거나 없는 실세계 조작에는 그대로 안 옮겨진다.
- **재구성 기반 표현**: 관측 **재구성**으로 잠재를 학습 → 태스크에 무관한 픽셀 디테일에 표현 용량을 낭비할 수 있다([Dreamer V1](dreamer.md)과 공유하는 한계).
- **장기 상상의 오차 누적**: 상상 지평이 길어지면 prior 오차가 복리로 쌓인다.
- **게임 벤치 중심**: Atari 55게임에 한정된 검증 — 실제 로봇 매니퓰레이션 등 실세계 적용은 별도 과제(→ 로봇 실용 최전선은 예측형 V-JEPA-2 쪽).

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

> 위 표의 값은 원문(arXiv 1912.01603 / 2010.02193 / 2301.04104)에서 대조 완료. 원형은 [Dreamer V1](dreamer.md), 이후 확장은 [DreamerV3](dreamerv3.md) 참고.
