# MuZero (2020)

*관측 재구성 없이 보상·가치·정책만 예측 + MCTS*

Schrittwieser, Antonoglou, Hubert, Simonyan, Sifre, Schmitt, Guez, Lockhart, Hassabis, Graepel, Lillicrap, Silver, *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero), Nature 2020 (arXiv 1911.08265).

## 한 줄 요약

**규칙(동역학)을 모르는** 환경에서 AlphaZero식 MCTS 계획을 하기 위해, 환경을 **재구성하지 않고** 계획에 필요한 것만 — **보상·가치·정책** — 예측하는 잠재 동역학 모델을 학습한다. Dreamer/PlaNet이 관측을 복원하는 것과 정반대 철학: **"계획에 쓸모없는 픽셀은 예측하지 않는다."**

---

## 1. 문제

AlphaZero는 강력하지만 **정확한 시뮬레이터(규칙)**를 요구한다 — 바둑·체스처럼 규칙이 알려진 곳에서만 된다. Atari 같은 환경은 규칙(동역학)을 모른다. 그렇다고 Dreamer식으로 **관측 전체를 재구성**하는 모델을 배우면, 계획엔 무관한 배경 픽셀까지 맞추느라 표현·계산을 낭비한다. 질문: **계획에 꼭 필요한 양만 예측하는 모델을 배울 수 있는가?**

## 2. 방법

### 2.1 세 함수 — representation / dynamics / prediction

MuZero는 관측을 **잠재 상태 $s$**로 넣고, 그 잠재공간 안에서만 미래를 굴린다. 세 신경망:

$$
\begin{aligned}
&\text{Representation:} && s^0 = h_\theta(o_1,\dots,o_t) &&\text{관측 이력} \to \text{루트 잠재}\\
&\text{Dynamics:} && s^{k}, r^{k} = g_\theta(s^{k-1}, a^{k}) &&\text{잠재 전이 + 즉시보상 예측}\\
&\text{Prediction:} && p^{k}, v^{k} = f_\theta(s^{k}) &&\text{정책 prior + 가치}
\end{aligned}
$$

핵심: **$s^k$는 관측을 복원할 의무가 없다.** 오직 $r, v, p$를 맞추는 데만 쓰이는 **추상 상태**다. 그래서 잠재는 "환경의 진짜 상태"일 필요가 없고, 계획에 유용한 통계량만 담으면 된다.

### 2.2 MCTS로 계획

각 결정 시점에서 루트 $s^0$부터 **Monte Carlo Tree Search**를 돌린다. 트리 노드는 잠재 상태, 엣지는 행동. 확장은 dynamics $g$로(실환경 없이 상상), 리프 평가는 prediction $f$의 $v$로. 선택은 AlphaZero의 **PUCT**:

$$
a^* = \arg\max_a\ \Big[\,Q(s,a) + P(s,a)\cdot\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}\cdot\Big(c_1 + \log\tfrac{\sum_b N(s,b)+c_2+1}{c_2}\Big)\Big],\quad c_1{=}1.25,\ c_2{=}19652
$$

- $Q(s,a)$: 그 엣지의 평균 가치(백업으로 갱신), $P(s,a)=p$: prediction의 정책 prior, $N$: 방문 수.
- 시뮬레이션 횟수(원문): **보드게임 800회/수, Atari 50회/수**(행동공간이 작아 50이면 충분). 시뮬레이션 후 루트의 **방문 분포 $\pi \propto N(s^0,\cdot)^{1/T}$**를 실제 행동 정책으로 사용.

MCTS가 곧 "학습된 모델 위에서의 계획"이고, 그 결과(방문 분포)가 정책망을 **개선하는 타깃**이 된다(정책 향상 연산자).

### 2.3 학습 — 무엇을 맞추나 (재구성이 없다)

리플레이의 실제 궤적에서, 모델을 $K$스텝 언롤해 세 가지를 실제 관측값에 정렬한다.

$$
\mathcal{L}(\theta) = \sum_{k=0}^{K}\Big[\underbrace{\ell^{r}(u_{t+k},\, r^{k})}_{\text{보상}} + \underbrace{\ell^{v}(z_{t+k},\, v^{k})}_{\text{가치}} + \underbrace{\ell^{p}(\pi_{t+k},\, p^{k})}_{\text{정책}}\Big] + c\|\theta\|^2
$$

- **보상 타깃** $u$: 환경에서 실제로 받은 보상.
- **가치 타깃** $z$: $n$-스텝 부트스트랩 return(Atari) 또는 게임 최종 결과(보드게임).
- **정책 타깃** $\pi$: 그 스텝 MCTS의 방문 분포.
- **관측 재구성 손실이 없다.** 잠재 $s^k$엔 아무 재구성 제약이 없고, 오직 위 세 예측이 맞도록 **간접적으로** 형성된다.
- 스칼라(보상·가치)는 넓은 동적 범위를 위해 **범주형(이산 support) + 가역 변환** $\;h(x)=\operatorname{sign}(x)\big(\sqrt{|x|+1}-1\big)+\varepsilon x,\ \varepsilon=0.001\;$로 스케일링해 회귀(Atari).

### 2.4 알고리즘 요약

```
Self-play(각 액터):
  매 스텝: s^0 = h(관측이력) → MCTS N_sim회(g로 확장, f로 평가, PUCT 선택)
           방문분포 π로 행동 샘플 → 환경 진행, 궤적 저장
Training(러너):
  리플레이에서 궤적 샘플 → 모델 K스텝 언롤
  보상·가치·정책 손실로 h,g,f 동시 학습 (재구성 없음)
```

## 3. 결과

| 도메인 | 결과 (원문) |
|--------|-------------|
| 바둑·체스·쇼기 | 규칙을 **주지 않고도** AlphaZero(규칙을 아는)의 초인 성능에 필적 |
| Atari 57게임 | human-normalized **중앙값 2041.1% / 평균 4999.2%** — 직전 SOTA R2D2(1920.6%/4024.9%)를 능가, **42/57게임 우위** |
| MuZero Reanalyze | 200M 프레임에서 중앙값 731.1% — 데이터 재사용 변형으로 샘플효율 대폭 개선 |

재구성 없는 학습 모델만으로 위 전부를 달성했다는 게 요점. 이후 **Sampled MuZero**(연속·큰 행동공간), **EfficientZero**(샘플효율)로 확장된다.

## 4. 스터디와의 개념적 연결

- **"계획에 필요한 것만 예측한다"**는 Dreamer/PlaNet의 재구성 철학과 대비되는 축. 재구성 없이(MuZero) / 표현 예측만으로(JEPA)라는 **재구성-free 갈래의 원형**이다.
- **task-relevant 표현**: 관측을 전부 복원하지 않고 보상·가치에 쓸모 있는 통계만 남긴다 — 로봇 인지에서 "집기에 필요한 것(pose·접촉점)만 뽑는다"는 관점과 정합. 픽셀 완벽 복원이 목적이 아니라는 점.
- 다만 **MCTS 계획은 이산·저차원 행동에서 강력**하고, 연속·고차원 매니퓰레이션엔 그대로는 무겁다(→ Sampled MuZero, 또는 Dreamer식 상각 정책). 로봇 실용성은 별도 고려.

## 5. 한 줄 평·한계

**한 줄 평.** "world model은 세계를 그리는 게 아니라, 계획에 쓸 통계를 예측하는 것"이라는 관점 전환. 재구성을 버려도(오히려 버려서) 계획이 된다는 걸 보인 이정표.

**한계.**
- **MCTS 비용**: 매 스텝 수십~수백 시뮬레이션. 실시간·고빈도 제어엔 부담(EfficientZero 등이 완화).
- **행동공간**: 원판은 이산 중심. 연속·고차원은 Sampled MuZero 필요.
- **해석성**: 잠재가 재구성 제약이 없어 **사람이 들여다볼 그림이 없다**(Dreamer는 디코더로 상상을 시각화 가능 — 트레이드오프).
- **보상 의존**: 여전히 보상 신호가 전제인 계획 프레임(모방·무보상 표현학습과는 다른 문제).
