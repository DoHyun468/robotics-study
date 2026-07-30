# V-JEPA / V-JEPA-2 — 픽셀이 아니라 표현공간에서 예측하는 world model

작성: world-models 심화 트랙 · 원문 기준
- V-JEPA: Bardes, Garrido, Assran 외 (Meta AI), *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA), arXiv 2404.08471, 2024.
- V-JEPA 2: Assran 외 (Meta AI), *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv 2506.09985, 2025. + action-conditioned 변형 V-JEPA 2-AC.
- 계보: LeCun의 JEPA(Joint-Embedding Predictive Architecture) 구상 → I-JEPA(이미지) → V-JEPA(영상) → V-JEPA 2(스케일 + 로봇 계획).

> V-JEPA 2 수치는 원문(2506.09985) 대조 완료. 원조 V-JEPA(v1)의 개별 벤치 절대값 일부는 미기재.

## 한 줄 요약

관측을 **재구성하지 않고**, 마스킹된 부분의 **표현(representation)**을 표현공간에서 예측하는 자기지도 world model. 픽셀 복원(Dreamer)도, 보상 예측(MuZero)도 아닌 **제3의 축(예측·표현형)**. V-JEPA 2는 이를 대규모로 키우고 **행동 조건(action-conditioned)**을 붙여 **실제 로봇 계획**에 사용 — 로봇 실용 최전선이자, 3D/기하 강점과 가장 가까운 갈래.

---

## 1. 문제

생성형 world model(픽셀 예측)은 화려하지만, **태스크에 무관한 디테일**(배경 질감, 정확한 색)까지 맞추느라 표현·계산을 낭비하고 장기적으로 흐릿·불안정해진다. 반대로 대비학습(contrastive)은 **음성 쌍(negative)**과 강한 증강에 민감하다. 질문: **재구성도, negative도 없이** 영상에서 예측적이고 태스크에 유용한 표현을 배울 수 있는가?

JEPA의 답: **입력공간(픽셀)이 아니라 표현공간에서 예측한다.** 예측 불가능한 디테일은 표현이 알아서 버리고, **예측 가능한 구조(물체·운동·기하)**만 남긴다.

## 2. 방법

### 2.1 JEPA 일반 구조

컨텍스트 $x$(보이는 부분)와 타깃 $y$(가려진 부분)를 둔다. 세 요소:

$$
\begin{aligned}
&\text{Context encoder:} && s_x = E_\theta(x)\\
&\text{Target encoder (EMA):} && s_y = E_{\bar\theta}(y),\quad \bar\theta \leftarrow \tau\bar\theta + (1-\tau)\theta \\
&\text{Predictor:} && \hat s_y = P_\phi(s_x,\, \Delta) &&\Delta=\text{타깃 위치·마스크 토큰}
\end{aligned}
$$

목적: **표현공간에서** 예측을 타깃 표현에 맞춘다(재구성 아님). 타깃엔 stop-gradient.

$$
\mathcal{L}(\theta,\phi) = \big\|\,P_\phi(E_\theta(x),\Delta) - \operatorname{sg}\big(E_{\bar\theta}(y)\big)\,\big\|_{1}
$$

**붕괴(collapse) 방지**: contrastive의 negative 대신 **비대칭 설계 + EMA target encoder + stop-grad**로 자명해($s\equiv$const)를 회피. (I-JEPA에서 확립된 레시피를 영상으로.)

### 2.2 V-JEPA — 영상으로 확장

- 입력은 비디오 클립. **시공간(spatiotemporal) 튜브를 마스킹**하고, 가려진 튜브의 표현을 컨텍스트로부터 예측.
- 픽셀 디코더 없음. 학습된 인코더 표현을 **frozen**으로 두고 downstream(행동 인식 등)에서 가벼운 probe만 붙여 평가 → 표현 품질을 직접 측정.
- 결과 표현이 **운동·상호작용** 같은 시간 구조를 담아, 이미지 자기지도보다 영상 이해 태스크에서 강함(구체 벤치 수치는 3절 참조).

### 2.3 V-JEPA 2 — 스케일 + 행동 조건 + 로봇 계획

- **스케일업**: **100만 시간 이상의 인터넷 영상+이미지**로 action-free 사전학습해 범용 시각 표현·예측 능력을 키움.
- **V-JEPA 2-AC (action-conditioned)**: 사전학습된 표현 위에, **행동을 조건으로 다음 표현을 예측**하는 잠재 world model을 얹는다 — Droid 데이터셋의 **라벨 없는 로봇 영상 62시간 미만**만으로 학습(적은 상호작용 데이터).

$$
\hat s_{t+1} = W_\phi(s_t,\, a_t),\qquad s_t = E(\text{관측}_t)
$$

- **계획(MPC)**: 이 표현공간 예측기로 후보 행동열을 굴려, 목표 표현 $s^\ast$(목표 이미지의 표현)에 가까워지는 행동을 고른다. 보상 라벨 없이 **목표 이미지 하나로** 조작을 계획(zero-shot에 가깝게 새 로봇·환경으로 이전).

$$
a^\ast = \arg\min_{a_{1:H}}\ \big\| \hat s_{t+H}(a_{1:H}) - s^\ast \big\|
$$

- 픽셀을 그리지 않고 **표현공간에서만** 미래를 굴려 계획한다는 게 Dreamer(픽셀 재구성 기반 상상)와의 결정적 차이.

## 3. 결과

- **V-JEPA (v1, 원문 보고)**: 재구성·contrastive 없이 **frozen** 표현만 평가. 최대 모델 **ViT-H/16(영상만 학습)** 기준 **Kinetics-400 81.9 / Something-Something-v2 72.2 / ImageNet-1K 77.9**. 영상 이해(운동·시간 구조)에서 특히 강함.
- **V-JEPA 2 (원문 보고)**: 동작 이해 **Something-Something v2 top-1 77.3**, 인간 행동 예측 **Epic-Kitchens-100 recall-at-5 39.7(SOTA)**. 로봇: **V-JEPA 2-AC를 서로 다른 두 실험실의 Franka 팔에 zero-shot 배치** — 해당 로봇에서 데이터 수집·태스크별 학습·보상 없이, **목표 이미지 기반 계획만으로** 픽앤플레이스 수행.

## 4. 스터디와의 개념적 연결 (가장 중요한 갈래)

- **우리 개인 방향과 정확히 일치**: "WM은 예측형(JEPA) 위주." 이 파일이 트랙의 실질 앵커 후보다.
- **3D/기하 강점과 최근접**: 표현공간에서 물체·운동·기하 구조를 예측한다는 건, 이 사이트에서 다룬 point cloud·pose·depth로 관측을 **기하 상태**로 요약하는 것과 같은 철학(태스크-무관 픽셀 버리기). V-JEPA의 "예측 가능한 구조만 남긴다"는 depth/registration 관점의 학습판.
- **목표 이미지 기반 계획**은 이 사이트의 "인지 → 타깃 → 도달" 파이프라인과 직접 대응: V-JEPA 2-AC의 $\arg\min\|\hat s_{t+H}-s^\ast\|$는, pose 타깃으로 IK를 풀던 것을 **표현공간 MPC**로 일반화한 형태로 읽을 수 있다.
- **보상 없이 계획**: 보상 설계가 어려운 실조작에서, 목표 표현만으로 움직인다는 점이 WM-RL(Dreamer/MuZero)의 보상 의존 한계를 보완.

## 5. 한 줄 평·한계

**한 줄 평.** "미래를 그리지 말고, 미래의 **표현**을 예측하라." 재구성의 낭비와 contrastive의 취약함을 동시에 피하는 제3의 길이며, V-JEPA 2가 이를 **실로봇 계획**까지 끌고 간 게 결정적. 로봇 실용 최전선.

**한계.**
- **표현공간 목표의 정의**: "목표 표현 $s^\ast$"를 무엇으로/얼마나 정밀하게 줄지가 태스크 성패를 가른다. 정밀 6-DoF·접촉 풍부 조작에서 표현 예측만으로 충분한지는 열린 문제(**미확인** 영역).
- **평가 난이도**: 재구성이 없어 "무엇을 예측하는지"가 픽셀로 안 보인다(MuZero와 공유하는 해석성 문제).
- **장기·물리 일관성**: 표현공간 예측도 지평이 길면 흐려진다. 장기 계획은 여전히 짧은 지평 MPC로 상각.
- **로봇 검증 범위**: zero-shot Franka 픽앤플레이스는 인상적이나, 접촉 풍부·정밀 6-DoF 조작으로의 확장은 후속 과제(원문도 픽앤플레이스류 중심).
