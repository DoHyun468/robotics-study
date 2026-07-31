# V-JEPA (2024)

*픽셀이 아니라 표현공간에서 예측하는 영상 자기지도 학습*

Bardes, Garrido, Assran 외 (Meta AI), *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA), arXiv 2404.08471, 2024.

계보: LeCun의 JEPA(Joint-Embedding Predictive Architecture) 구상 → I-JEPA(이미지) → **V-JEPA(영상)** → [V-JEPA 2](vjepa2.md)(스케일 + 로봇 계획).

## 한 줄 요약

관측을 **재구성하지 않고**, 마스킹된 부분의 **표현**(representation)을 표현공간에서 예측하는 자기지도 world model. 픽셀 복원(Dreamer)도, 보상 예측(MuZero)도 아닌 **제3의 축(예측·표현형)**. 같은 조건에서 픽셀 예측(VideoMAE)보다 정확도가 높고(**SSv2 69.5 vs 65.5**) 학습도 빠르다는 걸 보이며, "미래의 픽셀이 아니라 미래의 표현을 맞혀라"를 영상에서 입증한 원조편이다.

---

## 1. 문제 — 무엇을 예측 대상으로 삼을 것인가

영상 자기지도 학습의 두 주류엔 각자 아픈 데가 있다.

- **픽셀 재구성**(MAE/VideoMAE류): 가려진 픽셀을 복원한다. 직관적이지만, **예측 불가능하거나 태스크에 무관한 디테일**(배경 질감, 나뭇잎의 정확한 흔들림, 조명 노이즈)까지 맞추는 데 모델 용량과 계산을 쓴다. 세상엔 원리적으로 예측 불가능한 픽셀이 많다 — 그걸 맞추라고 강요하면 표현이 낭비된다.
- **대비학습**(contrastive): 픽셀은 안 그리지만, **음성 쌍**(negative)과 강한 증강 설계에 민감하고, 그 설계가 곧 귀납편향이 된다.

JEPA의 답은 셋째 길이다: **입력공간(픽셀)이 아니라 표현공간에서 예측한다.** 표현은 학습되는 것이므로, 예측 불가능한 디테일은 표현이 알아서 버리고 **예측 가능한 구조**(물체·운동·기하)만 남긴다. 예측 대상 자체가 "맞힐 가치가 있는 것"으로 스스로 정제되는 셈이다.

> 이 논문 제목이 "Revisiting **Feature Prediction**"인 이유: 질문이 정확히 "픽셀 예측 대신 특징 예측만으로 어디까지 가는가"다.

## 2. 방법

### 2.1 JEPA 일반 구조 — 세 요소

컨텍스트 $x$(보이는 부분)와 타깃 $y$(가려진 부분)를 두고:

$$
\begin{aligned}
&\text{Context encoder:} && s_x = E_\theta(x) && \text{보이는 부분을 인코딩}\\
&\text{Target encoder (EMA):} && s_y = E_{\bar\theta}(y),\quad \bar\theta \leftarrow \tau\bar\theta + (1-\tau)\theta && \text{타깃의 "정답 표현" 생성}\\
&\text{Predictor:} && \hat s_y = P_\phi(s_x,\, \Delta) && \Delta=\text{타깃 위치 정보(마스크 토큰)}
\end{aligned}
$$

여기서 $E_\theta$는 학습되는 인코더, $E_{\bar\theta}$는 그 **지수이동평균(EMA)** 복사본, $P_\phi$는 "컨텍스트 표현 + 어디를 맞힐지"를 받아 타깃 표현을 예측하는 작은 네트워크다. 손실은 표현공간의 L1 거리(타깃엔 stop-gradient):

$$
\mathcal{L}(\theta,\phi) = \big\|\,P_\phi(E_\theta(x),\Delta) - \operatorname{sg}\big(E_{\bar\theta}(y)\big)\,\big\|_{1}
$$

**붕괴(collapse)는 왜 안 일어나나?** 모든 입력에 같은 표현을 내면($s\equiv$const) 손실이 0이 되는 자명해가 존재한다. contrastive는 negative로 이걸 막지만, JEPA는 **비대칭 설계**(predictor는 컨텍스트 쪽에만) + **EMA target encoder** + **stop-gradient** 조합으로 막는다 — I-JEPA에서 확립된 레시피의 영상판이다.

### 2.2 영상 마스킹 — 시공간 튜브를 크게 가린다

이미지와 달리 영상은 인접 프레임 간 중복이 커서, 대충 가리면 "옆 프레임 베끼기"로 풀려버린다. V-JEPA의 마스킹(원문):

- **short-range 마스크**: 프레임당 15%를 덮는 블록 8개.
- **long-range 마스크**: 프레임당 70%를 덮는 블록 2개.
- 두 마스크 모두 공간 블록을 **전체 시간축으로 연장**(튜브 마스킹) — 시간으로 새는 지름길 차단.
- 합쳐서 **평균 마스킹 비율 ~90%** — 아주 조금만 보여주고 크게 맞히게 한다.

### 2.3 학습·평가 셋업

- **데이터**: VideoMix2M — HowTo100M + Kinetics-400/600/700 + SSv2를 합쳐 검증셋 중복을 제거한 **약 200만 영상**.
- **모델**: ViT-L/16, ViT-H/16, ViT-H/16(384) — 픽셀 디코더는 없다.
- **평가**: 인코더를 **frozen**으로 고정하고 **attentive probe**(cross-attention 1층 + MLP + linear)만 학습해 표현 품질을 직접 측정. "파인튜닝발"이 아니라 표현 자체의 힘을 재는 프로토콜이다.

## 3. 결과 (원문 대조)

**frozen 표현, ViT-H/16 기준:**

| 벤치 | 지표 | V-JEPA |
|---|---|---|
| Kinetics-400 (외형 중심) | top-1 | 81.9 |
| Something-Something-v2 (운동 중심) | top-1 | 72.2 |
| ImageNet-1K (이미지) | top-1 | 77.9 |

**픽셀 예측과의 직접 비교** (같은 조건, SSv2): V-JEPA **69.5** vs VideoMAE **65.5** — 특징 예측이 픽셀 예측을 +4%p 이긴다. 학습 wall-clock도 대형 픽셀 예측 모델 대비 **약 2배 빠름**(원문 Fig. 5). 특히 **운동(motion) 이해가 필요한 SSv2**에서 격차가 큰 것이 요점 — 시간 구조를 픽셀이 아닌 표현으로 예측한 효과다.

## 4. 스터디와의 개념적 연결

- "**예측 가능한 구조만 남긴다**"는 이 사이트 [perception 파이프라인](../perception.md)의 철학과 같다 — point cloud·pose·depth 추정도 결국 관측에서 태스크 무관 픽셀을 버리고 **기하 상태**만 남기는 일이다. V-JEPA는 그 선별을 손으로 설계하지 않고 학습으로 얻는다.
- **World model 3축** 중 예측·표현형의 원조. 재구성(Dreamer) — 재구성 없이 보상만(MuZero) — 표현만(JEPA)의 스펙트럼에서 가장 오른쪽. 반대편 극단([DIAMOND](diamond.md): 재구성을 더 잘하기)과 대비해 읽으면 구도가 선명하다. 축 정리는 [Concepts](../concepts.md).
- 여기엔 행동도 보상도 없다 — **순수 표현학습**이다. 행동 조건과 로봇 계획은 [V-JEPA 2](vjepa2.md)가 얹는다.

## 5. 한 줄 평·한계

**한 줄 평.** "미래를 그리지 말고, 미래의 **표현**을 예측하라"를 영상에서 처음 정면 검증한 논문. 픽셀 예측과의 통제 비교(69.5 vs 65.5, 2배 빠름)를 만들어준 덕에, 이후 JEPA 계열 주장의 실증 기반이 됐다.

**한계.**
- **frozen 평가의 한계**: probe로 잰 표현 품질이 실제 행동·계획으로 얼마나 전이되는지는 이 논문만으론 모른다 — 그 답이 [V-JEPA 2](vjepa2.md)다.
- **행동 조건 없음**: action-conditioned 예측·로봇 적용은 원조편 범위 밖.
- **해석성**: 재구성이 없어 "무엇을 예측하는지" 픽셀로 확인할 수 없다(MuZero와 공유하는 트레이드오프).
- **외형 벤치에선 격차 축소**: 운동 중심(SSv2)에서 강점이 크고, 외형 중심(K400)·이미지(IN1K)에선 이미지 자기지도 대비 우위가 작다 — 시간 구조 예측이라는 설계 의도 그대로의 프로파일.
