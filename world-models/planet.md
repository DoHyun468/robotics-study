# PlaNet (2019)

*잠재 동역학 + 온라인 계획(정책망 없이)*

Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, Davidson, *Learning Latent Dynamics for Planning from Pixels* (PlaNet), ICML 2019 (arXiv 1811.04551).

## 한 줄 요약

이미지에서 **저차원 잠재 동역학(RSSM)**을 학습하고, 정책망 없이 매 스텝 **CEM(교차 엔트로피 계획)**으로 잠재공간에서 미래를 굴려 행동을 고른다. Dreamer의 **직접 전신** — RSSM과 "픽셀 대신 잠재에서 계획"이라는 뼈대는 여기서 나왔고, Dreamer는 여기에 **학습된 정책·가치망**을 얹어 계획을 상각(amortize)한 것이다.

---

## 1. 문제

이미지 기반 연속제어에서 model-free RL은 샘플이 너무 많이 든다. Model-based로 가면 샘플효율은 오르지만, **픽셀 공간에서 미래를 예측**하면 (a) 계산이 비싸고 (b) 고차원 예측 오차가 계획을 망친다. 핵심 질문: **미래 예측을 어디서 할 것인가?** PlaNet의 답 — 픽셀이 아니라 **컴팩트한 잠재 상태 공간**에서.

## 2. 방법

### 2.1 RSSM (여기서 처음 제안)

잠재를 결정적 $h_t$(GRU) + 확률적 $s_t$(가우시안)으로 분리하는 **Recurrent State-Space Model**이 이 논문의 핵심 기여다(Dreamer가 그대로 계승).

$$
\begin{aligned}
&\text{Deterministic:} && h_t = f(h_{t-1}, s_{t-1}, a_{t-1}) &&(\text{GRU})\\
&\text{Transition (prior):} && s_t \sim p(s_t \mid h_t) \\
&\text{Posterior:} && s_t \sim q(s_t \mid h_t, o_t) \\
&\text{Observation decoder:} && o_t \sim p(o_t \mid h_t, s_t) \\
&\text{Reward:} && r_t \sim p(r_t \mid h_t, s_t)
\end{aligned}
$$

**설계 논지:** 순수 확률(stochastic-only) 모델은 정보를 여러 스텝 나르기 어렵고, 순수 결정적(RNN-only) 모델은 다봉 미래를 표현 못 한다. **둘을 합쳐야** 장기 정보 전달과 불확실성 표현을 동시에 얻는다 — RSSM의 존재 이유.

### 2.2 학습 손실 (ELBO)

$$
\mathcal{J} = \mathbb{E}_q\!\Big[\sum_t \ln p(o_t\mid h_t,s_t) + \ln p(r_t\mid h_t,s_t) - \mathrm{KL}\big[q(s_t\mid h_t,o_t)\,\|\,p(s_t\mid h_t)\big]\Big]
$$

Dreamer의 world model 손실과 사실상 동일(재구성 + prior/posterior KL).

### 2.3 Latent overshooting (PlaNet 고유 정규화)

한 스텝 KL만으로는 다중 스텝 예측이 부정확해질 수 있어, **여러 스텝 앞의 prior**를 데이터로 정렬하는 정규화를 추가한다. $d$-스텝 앞까지의 예측 분포를 posterior로 끌어당긴다.

$$
\mathcal{J}_{\text{overshoot}} = -\sum_{d} \beta_d\; \mathbb{E}\big[\mathrm{KL}[\,q_{t}\,\|\,p_{t\mid t-d}\,]\big]
$$

(단일 스텝 정렬만 하면 장기 rollout에서 오차가 커지는 걸 막는 장치. Dreamer는 이 항 대신 상상 학습으로 우회.)

### 2.4 계획 — CEM (정책망 없음)

PlaNet의 결정적 차이. **정책을 학습하지 않고**, 매 결정 시점마다 잠재 동역학 위에서 **모델 예측 제어(MPC)**를 CEM으로 돈다.

```
매 스텝 t (현재 posterior 상태 s_t 에서):
  가우시안 행동 시퀀스 분포 N(μ, σ²) 초기화 (길이 H=12)
  반복 I=10회:
    J=1000개 후보 행동 시퀀스 샘플
    각 후보를 prior로 H스텝 잠재 rollout → 예측 보상 합 Σ r̂
    상위 K=100개(elite) 선택 → 그 평균/분산으로 μ, σ 갱신
  μ의 첫 행동 a_t 실행 (나머지는 버림, receding horizon)
```

원문 CEM 설정(Appendix A): **H=12, I=10, J=1000, K=100**. 정책망 없이 매 스텝 이 최적화를 처음부터 다시 돈다.

- **장점:** 학습할 정책 파라미터가 없어 구현이 단순하고, 보상·목표가 바뀌어도 재학습 없이 대응(계획이 즉석에서 최적화).
- **단점:** **매 스텝 수백 rollout**을 돌려 느리다. 계획 지평 $H$ 밖은 못 본다(근시안). 이 두 약점이 Dreamer가 정책·가치망을 도입한 직접 동기.

## 3. 결과

이미지 입력 연속제어 **DeepMind Control Suite 6개 태스크**(cartpole swingup, reacher easy, cheetah run, finger spin, cup catch, walker walk).

| 비교 대상 | 결과 (원문) |
|-----------|-------------|
| A3C (proprioceptive 상태 입력) | PlaNet이 **100 에피소드 안에**, 10만 에피소드 학습한 A3C를 **전 태스크에서 능가** |
| D4PG (이미지 입력) | 태스크에 따라 **약 100~500배 적은 에피소드**로 유사 성능 |
| 하이퍼파라미터 | 단일 에이전트·단일 설정으로 6태스크 모두 처리 |

"이미지만 보고, 상태를 직접 받은 model-free보다 1,000배 적은 데이터로 이긴다"가 이 논문의 헤드라인이다 — 잠재공간 계획이라는 방향의 유효성을 처음 크게 보인 수치.

## 4. 스터디와의 개념적 연결

- **RSSM의 원산지**. Dreamer를 이해하려면 여기의 결정적+확률적 분리와 latent overshooting을 먼저 아는 게 정공법.
- **MPC(CEM) 계획**은 우리가 로봇 파이프라인에서 다룬 "타깃 → IK/제어"의 상위 개념(행동 시퀀스 최적화)과 같은 계열. PlaNet은 그 최적화를 **학습된 잠재 동역학** 위에서 한다는 점이 다르다.
- "미래 예측을 저차원 공간에서" — 우리 depth/3D 강점이 관측을 **기하 상태**로 압축하는 것과 같은 동기(고차원 픽셀을 직접 다루지 않는다).

## 5. 한 줄 평·한계

**한 줄 평.** "픽셀 대신 잠재에서 굴린다"의 원형. RSSM 하나로 이후 Dreamer 계보 전체가 파생됐다는 점에서 계보의 뿌리.

**한계.** (1) 매 스텝 온라인 계획이라 **느리고 근시안적** — Dreamer가 정책/가치로 해결. (2) 재구성 기반 표현의 한계는 Dreamer와 공유. (3) 이산 행동·복잡 동역학(Atari 류)엔 약함 — 가우시안 잠재의 한계로, V2의 이산 잠재가 답.
