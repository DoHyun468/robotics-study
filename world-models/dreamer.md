# Dreamer V1/V2/V3 — 잠재 상상(latent imagination)에서 배우는 행동

작성: world-models 심화 트랙 · 원문 3편 기준
- V1: Hafner, Lillicrap, Ba, Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination*, ICLR 2020.
- V2: Hafner, Lillicrap, Norouzi, Ba, *Mastering Atari with Discrete World Models*, ICLR 2021.
- V3: Hafner, Pasukonis, Ba, Lillicrap, *Mastering Diverse Domains through World Models*, 2023 (arXiv 2301.04104).

## 한 줄 요약

관측을 **잠재 상태(latent state)**로 압축하는 world model(RSSM)을 배우고, **실제 환경이 아니라 그 모델 안(=상상)**에서 actor-critic으로 행동을 학습한다. 실환경 접촉을 최소화해 샘플효율이 높고, V1(연속제어)→V2(Atari 인간급)→V3(하이퍼파라미터 하나로 150+ 태스크·Minecraft 다이아)로 **같은 골격을 유지한 채 견고성만 끌어올려** 도메인 일반성을 얻었다.

---

## 1. 문제

시각 제어를 **POMDP**로 둔다. 에이전트는 진짜 상태 $s_t$(속도·물체 위치 등)를 직접 못 보고 관측 $o_t$(이미지)만 받는다. 그래서 행동은 **이력 전체**에 의존해야 한다.

$$
a_t \sim p(a_t \mid o_{\le t}, a_{<t}), \qquad o_t, r_t \sim p(o_t, r_t \mid o_{<t}, a_{<t}), \qquad \max\ \mathbb{E}_p\!\Big[\textstyle\sum_{t=1}^{T} r_t\Big]
$$

두 갈래의 해법이 있다.

- **Model-free RL** (DQN·PPO·SAC 등): 환경에서 직접 정책을 배운다. 단순하지만 **샘플 비효율**(수천만~수억 프레임). 실제 로봇엔 치명적.
- **Model-based RL**: 환경의 동역학을 모델로 배우고 그 안에서 계획/학습. 샘플효율이 높지만, **픽셀 공간에서 미래를 굴리면** 비싸고 오차가 누적된다.

Dreamer의 baseline인 **PlaNet**은 이 딜레마를 "픽셀이 아니라 **저차원 잠재공간**에서 미래를 굴린다"로 풀었다. 다만 PlaNet은 정책망 없이 매 스텝 **CEM 온라인 계획**을 돌려 느리고 근시안적이었다. Dreamer의 기여는 **정책·가치망을 잠재 상상 궤적 위에서 학습**해, 계획을 학습된 정책으로 상각(amortize)한 것이다.

> 왜 "상상에서 학습"이 이득인가: 일단 world model이 있으면, 실데이터 1스텝당 잠재공간에서 수십~수백 스텝의 가상 궤적을 값싸게 생성할 수 있다. 정책은 이 가상 데이터로 갱신되므로 **실환경 상호작용당 정책 개선량이 크다** = 샘플효율.

---

## 2. 방법

### 2.1 RSSM — Recurrent State-Space Model

잠재 상태를 두 부분으로 나눈다. 이게 RSSM의 핵심 설계다.

- **결정적(deterministic)** $h_t$: GRU로 이어지는 기억. 장기 의존성·경로를 안정적으로 나른다.
- **확률적(stochastic)** $z_t$: 불확실성을 표현. V1은 대각 가우시안, V2/V3은 범주형(categorical).

모델 상태(model state)는 둘의 결합 $\;\text{state}_t = (h_t, z_t)$. 구성 요소:

$$
\begin{aligned}
&\text{Recurrent (sequence) model:} && h_t = f_\phi(h_{t-1},\, z_{t-1},\, a_{t-1}) &&(\text{GRU})\\
&\text{Representation model (posterior):} && z_t \sim q_\phi(z_t \mid h_t,\, x_t) &&(x_t=\text{관측})\\
&\text{Transition model (prior):} && \hat z_t \sim p_\phi(\hat z_t \mid h_t) \\
&\text{Decoder (image/vector):} && \hat x_t \sim p_\phi(\hat x_t \mid h_t, z_t) \\
&\text{Reward predictor:} && \hat r_t \sim p_\phi(\hat r_t \mid h_t, z_t) \\
&\text{Continue predictor (V2+):} && \hat c_t \sim p_\phi(\hat c_t \mid h_t, z_t) &&(c_t\in\{0,1\}\ \text{에피소드 지속})
\end{aligned}
$$

**posterior vs prior**가 이 모델의 심장이다.

- **posterior $q(z_t\mid h_t, x_t)$**: 관측을 **보고** 상태를 추론.
- **prior $p(\hat z_t\mid h_t)$**: 관측 **없이** 다음 상태를 예측. → **상상(imagination) 때 쓰는 게 이것**. 관측 없이 미래를 굴려야 하니까.

학습은 이 둘을 가깝게(KL) 만들어, 관측이 없어도 prior만으로 정확히 미래를 예측하도록 강제한다.

### 2.2 World model 손실 — variational lower bound

관측·보상 재구성 + prior/posterior 정렬. V1의 형태(ELBO):

$$
\mathcal{J}_{\text{model}}(\phi) = \mathbb{E}_{q_\phi}\!\Bigg[\sum_t \underbrace{\ln p_\phi(x_t\mid h_t,z_t)}_{\text{관측 복원}} + \underbrace{\ln p_\phi(r_t\mid h_t,z_t)}_{\text{보상 예측}} - \underbrace{\beta\,\mathrm{KL}\!\big[q_\phi(z_t\mid h_t,x_t)\,\big\|\,p_\phi(z_t\mid h_t)\big]}_{\text{prior↔posterior 정렬}}\Bigg]
$$

- 앞 두 항: 이미지·보상을 잘 맞춰라(디코더가 잠재에 정보를 담게 강제).
- KL 항: **관측 없이 예측하는 prior가, 관측 본 posterior를 따라오게** 하라. 이게 곧 "동역학이 정확하다"는 뜻.
- V1은 KL이 0으로 붕괴하는 걸 막으려 **free nats**(KL을 일정 값 이하로는 벌하지 않음, 3 nats)를 쓴다.

> V2/V3에서 이 KL이 **KL balancing**과 **free bits(1 nat)**로 바뀐다(2.5·2.6절).

### 2.3 잠재 상상에서의 Actor-Critic (Dreamer의 진짜 기여)

실데이터 배치에서 얻은 각 model state $\{(h_t,z_t)\}$를 **시작점**으로, world model의 **prior**로 길이 $H$(보통 15~16)의 **상상 궤적**을 생성한다(실환경 접촉 0):

$$
\hat z_\tau \sim p_\phi(\hat z_\tau\mid h_\tau),\qquad a_\tau \sim \pi_\theta(a_\tau\mid h_\tau,\hat z_\tau),\qquad h_{\tau+1}=f_\phi(h_\tau,\hat z_\tau,a_\tau)
$$

두 네트워크를 학습한다.

$$
\begin{aligned}
&\text{Actor (action model):} && a_\tau \sim \pi_\theta(a_\tau\mid h_\tau, z_\tau)\\
&\text{Critic (value model):} && v_\psi(h_\tau,z_\tau) \approx \mathbb{E}_{\pi_\theta}\!\Big[\textstyle\sum_{k\ge\tau}\gamma^{\,k-\tau} r_k\Big]
\end{aligned}
$$

**λ-return**으로 각 상상 지점의 목표 가치를 만든다(bias–variance 절충; TD(λ)의 model-based 버전):

$$
V^\lambda_\tau = \hat r_\tau + \hat\gamma_\tau\Big[(1-\lambda)\,v_\psi(h_{\tau+1},z_{\tau+1}) + \lambda\,V^\lambda_{\tau+1}\Big],\qquad V^\lambda_H = v_\psi(h_H,z_H)
$$

($\hat\gamma_\tau$는 continue predictor로 예측한 지속 확률 $\hat c_\tau$에 할인 $\gamma$를 곱한 것; 에피소드 종료를 상상 안에서 처리.)

두 목적함수:

$$
\max_\theta\ \mathbb{E}\Big[\textstyle\sum_\tau V^\lambda_\tau\Big]
\qquad\qquad
\min_\psi\ \mathbb{E}\Big[\textstyle\sum_\tau \tfrac12\big\|v_\psi(h_\tau,z_\tau) - \operatorname{sg}(V^\lambda_\tau)\big\|^2\Big]
$$

($\operatorname{sg}$ = stop-gradient. critic은 자기 타깃으로 회귀; V2+는 EMA target critic 사용.)

**Actor 그래디언트 — 여기가 버전별로 갈린다.**

- **V1(연속제어)**: world model·reward·value가 전부 미분 가능한 신경망이고 가우시안 잠재는 **reparameterization**으로 미분 가능하다. 따라서 actor 그래디언트를 **상상 궤적을 따라 직접 backprop**한다(analytic/pathwise gradient). model-free의 policy gradient가 겪는 높은 분산을 회피 — 이게 V1 샘플효율의 핵심.

$$
\nabla_\theta\, \mathbb{E}\Big[\textstyle\sum_\tau V^\lambda_\tau\Big]\quad\text{를 dynamics }f_\phi\text{를 관통해 계산 (pathwise)}
$$

- **V2(이산 행동, Atari)**: 이산 행동엔 pathwise가 안 통하므로 **REINFORCE**(score-function) + 엔트로피 정규화를 쓴다. 연속 행동엔 straight-through/pathwise. 실제 구현은 두 항을 혼합.

$$
\nabla_\theta \approx \mathbb{E}\Big[\sum_\tau \ln\pi_\theta(a_\tau)\,\operatorname{sg}(V^\lambda_\tau - v_\psi) \;+\; \eta\,\nabla_\theta\mathrm{H}[\pi_\theta(\cdot\mid h_\tau,z_\tau)]\Big]
$$

- **V3**: 원문 명시 — **연속 행동엔 stochastic backpropagation(pathwise), 이산 행동엔 REINFORCE**를 쓴다("stochastic backpropagation for continuous actions and reinforce for discrete actions"). 여기에 정규화된 return(2.6절)을 결합해 도메인 불문 하나의 하이퍼로 돌린다.

### 2.4 알고리즘 (전체 루프)

```
초기화: world model φ, actor θ, critic ψ, 리플레이 버퍼 D
반복:
  # (A) 실환경에서 데이터 수집
  o_1 관측
  for t = 1..T:
      h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1});  z_t ~ q_φ(z_t | h_t, o_t)   # posterior 인코딩
      a_t ~ π_θ(a_t | h_t, z_t)  (+탐험 노이즈);  환경에 a_t → o_{t+1}, r_t
  궤적을 D에 추가

  # (B) World model 학습
  D에서 시퀀스 배치 샘플
  posterior q로 인코딩 → J_model(φ) 로 φ 갱신
     (관측·보상·continue 복원 + KL[posterior‖prior])

  # (C) 상상에서 behavior 학습  ← 실환경 접촉 없음
  배치의 각 (h_t,z_t)를 시작점으로 prior p로 H스텝 롤아웃
     각 스텝 행동은 a_τ ~ π_θ, 보상·continue·value 예측
  λ-return V^λ 계산
  actor θ ← ∇ Σ V^λ 최대화 ;  critic ψ ← V^λ 회귀
```

세 과정(A 수집 / B 모델학습 / C 상상학습)이 번갈아 돈다. 데이터 효율의 원천은 **(A) 한 번에 (B),(C)를 여러 번** 돌릴 수 있다는 점.

### 2.5 V1 → V2: 이산 잠재 + KL balancing (Atari 정복의 열쇠)

V2는 **world model만 잠재공간에서 학습**해 Atari에서 인간급을 처음 달성했다. 바뀐 핵심 3가지:

**(1) 이산(categorical) 잠재.** $z_t$를 가우시안 대신 **32개 범주형 변수 × 각 32 클래스**의 벡터로 둔다(one-hot 32×32). 샘플링은 미분 불가라 **straight-through gradient**를 쓴다: forward는 one-hot 샘플, backward는 확률(softmax)로 흘린다.

$$
z = \operatorname{onehot}(\arg\max)\quad\text{(forward)},\qquad \nabla \leftarrow \nabla\,\text{softmax logits}\quad\text{(straight-through backward)}
$$

이산 잠재가 Atari 같은 다봉·불연속 동역학(장면 전환, 점수 점프)에 가우시안보다 잘 맞았다는 게 경험적 발견.

**(2) KL balancing.** KL 항을 prior/posterior 두 방향으로 나눠 **서로 다른 학습률**을 준다. prior를 posterior로 끌어당기는 쪽(동역학 학습)을 더 세게, 반대는 약하게.

$$
\mathcal{L}_{\mathrm{KL}} = \alpha\,\mathrm{KL}\!\big[\operatorname{sg}(q)\,\|\,p\big] + (1-\alpha)\,\mathrm{KL}\!\big[q\,\|\,\operatorname{sg}(p)\big],\qquad \alpha=0.8
$$

앞 항(prior가 움직임)에 가중치를 크게. representation이 prior를 맞추려 정보를 버리는 붕괴를 막는다.

**(3) 이산 행동 actor = REINFORCE + entropy**(2.3절). 그 외 continue predictor로 종료를 명시 모델링.

결과적으로 **단일 GPU, Atari 200M 프레임 예산**에서 Rainbow·IQN 등 강한 model-free를 같은 예산 내에서 능가.

### 2.6 V2 → V3: 견고성 기법 총동원 → "하이퍼 하나로 아무 도메인"

V3의 목표는 성능 SOTA 자체보다 **튜닝 없이 어떤 도메인이든** 도는 것. 도메인마다 보상 스케일·관측 분포가 극단적으로 다른데도 **고정 하이퍼파라미터**로 돌게 만든 기법들:

**(1) symlog 예측.** 보상·가치·벡터 관측처럼 스케일이 제각각인 타깃을 압착 함수로 정규화.

$$
\operatorname{symlog}(x) = \operatorname{sign}(x)\,\ln(1+|x|),\qquad \operatorname{symexp}(x)=\operatorname{sign}(x)\,(e^{|x|}-1)
$$

네트워크는 symlog 공간에서 예측하고 출력 시 symexp로 되돌린다. 큰 값의 gradient 폭발과 작은 값의 무시를 동시에 완화.

**(2) twohot 이산 회귀.** 스칼라(보상·가치)를 고정 bin 격자 위 **두 개의 이웃 bin에 선형 분배한 soft 분포**로 회귀 → 분류 손실로 학습. 회귀보다 넓은 동적 범위에 안정적.

$$
y \approx \sum_i p_i\, b_i,\qquad p\ \text{는 } y\text{를 감싸는 두 bin에만 질량을 두는 twohot 벡터}
$$

**(3) free bits + KL balancing.** dynamics/representation 두 KL을 나누고 각각 **1 nat 아래로는 벌하지 않는다**(free bits). 작은 KL 과최적화로 인한 표현 붕괴 방지.

$$
\mathcal{L}_{\text{dyn}} = \max\!\big(1,\ \mathrm{KL}[\operatorname{sg}(q)\,\|\,p]\big),\qquad
\mathcal{L}_{\text{rep}} = \max\!\big(1,\ \mathrm{KL}[q\,\|\,\operatorname{sg}(p)]\big)
$$

전체 world model 손실(가중치 원문 값):

$$
\mathcal{L}(\phi) = \mathbb{E}\Big[\sum_t \beta_{\text{pred}}\mathcal{L}_{\text{pred}} + \beta_{\text{dyn}}\mathcal{L}_{\text{dyn}} + \beta_{\text{rep}}\mathcal{L}_{\text{rep}}\Big],\quad \beta_{\text{pred}}{=}1,\ \beta_{\text{dyn}}{=}0.5,\ \beta_{\text{rep}}{=}0.1
$$

($\mathcal{L}_{\text{pred}}$ = 이미지/벡터 디코더 + 보상 + continue의 음의 로그가능도.)

**(4) unimix categoricals.** 범주형 확률에 **1% 균등분포를 섞어** 0 확률로 인한 KL 발산·죽은 클래스를 방지.

$$
p = 0.99\,\operatorname{softmax}(\text{logits}) + 0.01\,\tfrac{1}{K}
$$

**(5) percentile return normalization (actor).** 상상 return을 배치의 **5~95 백분위 폭 $S$**로 나눠 스케일을 자동 정규화(단, $\max(1,S)$로 하한). 보상 크기가 도메인마다 달라도 actor 스텝 크기가 일정해진다.

$$
\tilde V^\lambda_\tau = \frac{V^\lambda_\tau}{\max\!\big(1,\ \operatorname{Per}_{95}(V^\lambda)-\operatorname{Per}_{5}(V^\lambda)\big)}
$$

actor는 $\tilde V^\lambda$를 최대화 + **고정 엔트로피 계수**로 탐험 유지.

**(6) 스케일링.** 모델 크기를 키우면 성능·샘플효율이 **단조 상승**(작은 모델도 되지만 클수록 좋음) — 하나의 레시피로 크기만 키우면 됨을 보임.

이 기법들의 순효과: **DMC(연속·이미지/상태)·Atari·ProcGen·Crafter·Minecraft** 등 8개 도메인 150+ 태스크를 **동일 하이퍼**로 커버하고, **Minecraft 다이아몬드를 사람 데이터·커리큘럼 없이** 처음 채굴.

---

## 3. 결과

수치는 원문(arXiv 1912.01603 / 2010.02193 / 2301.04104) 대조 완료.

**V1 (연속제어, DeepMind Control Suite).** 이미지 입력 연속제어 20개 태스크에서 당시 model-based(PlaNet)·model-free(D4PG, A3C) 대비 **데이터 효율과 최종 성능 모두 우위**. 상상 지평 $H\approx15$, λ-return이 순수 가치 부트스트랩보다 안정적임을 ablation으로 확인(정확한 태스크별 점수는 원문 표 — 여기선 미기재).

**V2 (Atari 55게임, 200M 프레임).** world model을 픽셀 재구성으로 학습하고 **행동은 전부 잠재 상상에서** 배운 최초 에이전트로, **단일 GPU**에서 top 단일-GPU 에이전트 **IQN·Rainbow를 동일 연산예산·wall-clock에서 능가**. 원문 보고 점수(200M 스텝): **gamer-normalized 중앙값 2.15, 평균 11.33**(권장 지표인 clipped record mean은 0.28). 이산 행동 actor는 **순수 REINFORCE**(ρ=1, 엔트로피 계수 η=10⁻³), KL balancing α=0.8.

**V3 (다도메인 일반성).** **하나의 하이퍼파라미터 세트**로 8개 도메인 150+ 태스크에서 강력한 성능. 하이라이트: **Minecraft 다이아몬드를 from scratch(사람 데이터·커리큘럼 없이) 최초 획득**. 모델 크기 스케일업이 성능을 단조 개선. (개별 벤치 절대 점수는 원문 부록 — 미기재.)

---

## 4. 스터디와의 개념적 연결

- **관측 → 잠재 상태 압축**은 우리 파이프라인의 "인지 → 상태 요약"과 같은 축이다. RSSM의 posterior는 사실상 관측을 **저차원 표현**으로 인코딩하는 것으로, 우리가 다룬 point cloud/pose 추정이 관측을 **기하 상태**로 요약하는 것과 개념적으로 대응한다.
- **샘플효율(상상에서 학습)**은 실제 로봇 학습의 핵심 제약과 직결. 이 사이트의 [perception](../perception.md) 파이프라인에서 확인한 "실환경 데이터가 비싸다"는 문제의 정공법 중 하나가 world model 기반 상상 학습이다.
- **prior/posterior KL = 동역학의 예측가능성**. 관측 없이 prior만으로 미래를 굴릴 수 있어야 한다는 요구는, 우리 markerless ICP에서 "부분 관측만으로 pose를 복원"하려던 관측성(observability) 논의와 같은 결의 문제다.
- **주의(개인 방향과의 정합)**: 우리 트랙 개인 방향은 "RL은 언어 수준 이해까지, WM은 예측형(JEPA) 위주"다. Dreamer는 **보상 기반 WM-RL** 계열이라, 예측형(V-JEPA)과는 신호가 다르다(보상 vs 자기지도 표현). 계보의 뿌리로서 이해하되, 직접 실습 앵커로 삼진 않는다.

---

## 5. 한 줄 평·한계

**한 줄 평.** "픽셀 대신 잠재에서 미래를 굴리고, 실환경 대신 상상에서 정책을 배운다"는 하나의 아이디어를, 견고성 기법을 누적하며 V1→V3로 **골격을 안 바꾸고 일반성만 확장**한 모범적 계보. model-based RL이 튜닝 지옥을 벗어나 "레시피"가 될 수 있음을 V3가 보였다.

**한계.**
- **보상 신호 의존**: WM-RL은 보상이 정의된 태스크가 전제. 보상 설계가 어렵거나 없는 실세계 조작에는 그대로 안 옮겨진다(→ 모방/자기지도 표현형이 보완).
- **재구성 기반 표현**: V1/V2/V3 모두 관측 **재구성**으로 잠재를 학습 → 시각적으로 화려하지만 태스크에 무관한 픽셀 디테일에 표현 용량을 낭비할 수 있다. 이 지점을 **재구성 없이**(MuZero) 또는 **표현 예측만으로**(JEPA) 우회하는 갈래가 이후 흐름.
- **장기 상상의 오차 누적**: 상상 지평 $H$가 길어지면 prior 오차가 복리로 쌓인다. λ-return·짧은 $H$로 완화하지만 근본 해결은 아니다.
- **실로봇 검증은 제한적**: 벤치(게임·시뮬 제어) 중심. 실제 매니퓰레이션·접촉 풍부 태스크로의 이식은 별도 과제(→ 로봇 실용 최전선은 예측형 V-JEPA-2 쪽).

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

> 위 표의 값은 원문(arXiv 1912.01603 / 2010.02193 / 2301.04104)에서 대조 완료. V3 하이퍼(β_pred=1, β_dyn=0.5, β_rep=0.1, free bits 1 nat, unimix 1%, 5–95 백분위), V2(32×32 잠재, α=0.8, median 2.15/mean 11.33)는 본문·부록 기준.
