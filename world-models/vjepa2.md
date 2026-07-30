# V-JEPA 2 (2025)

작성: world-models 심화 트랙 · 원문 기준
- Assran 외 (Meta AI), *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv 2506.09985, 2025. + action-conditioned 변형 V-JEPA 2-AC.
- 계보: LeCun의 JEPA(Joint-Embedding Predictive Architecture) 구상 → I-JEPA(이미지) → [V-JEPA](vjepa.md)(영상, 2024) → **V-JEPA 2(스케일 + 로봇 계획, 2025)**.

> V-JEPA 2 수치는 원문(2506.09985) 대조 완료. 표현공간 예측이라는 기본 틀은 [V-JEPA](vjepa.md)와 동일 — 차이는 스케일·행동 조건·로봇 계획.

## 한 줄 요약

[V-JEPA](vjepa.md)가 확립한 "픽셀이 아니라 표현공간에서 예측한다"는 원칙을 대규모로 키우고, 여기에 **행동 조건(action-conditioned)**을 붙여 **실제 로봇 계획**에 사용한 후속작. 로봇 실용 최전선이자, 3D/기하 강점과 가장 가까운 갈래.

---

## 1. 문제

[V-JEPA](vjepa.md)는 재구성·negative 없이 표현공간 예측만으로 강한 영상 표현을 배울 수 있음을 보였다. 남은 질문 두 가지: (1) 이 표현을 **인터넷 규모**로 더 키우면 어디까지 가는가, (2) 표현공간 예측기를 **행동에 조건화**해 실제 로봇 조작 **계획**에 쓸 수 있는가 — 특히 보상 라벨도, 대량의 로봇 상호작용 데이터도 없이.

## 2. 방법

### 2.1 JEPA 배경 (간단 요약)

컨텍스트 인코더 $E_\theta$, EMA target 인코더 $E_{\bar\theta}$, 예측기 $P_\phi$로 구성해 표현공간에서 예측한다(재구성 아님, 타깃엔 stop-gradient):

$$
\mathcal{L}(\theta,\phi) = \big\|\,P_\phi(E_\theta(x),\Delta) - \operatorname{sg}\big(E_{\bar\theta}(y)\big)\,\big\|_{1}
$$

붕괴(collapse) 방지는 contrastive negative 대신 **비대칭 설계 + EMA target encoder + stop-grad**로. 자세한 유도는 [V-JEPA](vjepa.md) 참고.

### 2.2 스케일업 — action-free 사전학습

**100만 시간 이상의 인터넷 영상+이미지**로 action-free 사전학습해 범용 시각 표현·예측 능력을 키움. V-JEPA(v1)의 시공간 마스킹·표현예측 레시피를 그대로 규모만 확장.

### 2.3 V-JEPA 2-AC (action-conditioned) — 로봇 계획

- 사전학습된 표현 위에, **행동을 조건으로 다음 표현을 예측**하는 잠재 world model을 얹는다 — Droid 데이터셋의 **라벨 없는 로봇 영상 62시간 미만**만으로 학습(적은 상호작용 데이터).

$$
\hat s_{t+1} = W_\phi(s_t,\, a_t),\qquad s_t = E(\text{관측}_t)
$$

- **계획(MPC)**: 이 표현공간 예측기로 후보 행동열을 굴려, 목표 표현 $s^\ast$(목표 이미지의 표현)에 가까워지는 행동을 고른다. 보상 라벨 없이 **목표 이미지 하나로** 조작을 계획(zero-shot에 가깝게 새 로봇·환경으로 이전).

$$
a^\ast = \arg\min_{a_{1:H}}\ \big\| \hat s_{t+H}(a_{1:H}) - s^\ast \big\|
$$

- 픽셀을 그리지 않고 **표현공간에서만** 미래를 굴려 계획한다는 게 Dreamer(픽셀 재구성 기반 상상)와의 결정적 차이.

## 3. 결과

**V-JEPA 2 (원문 보고)**:

| 항목 | 값 |
|---|---|
| 사전학습 데이터 | 인터넷 영상+이미지 100만 시간 이상 |
| Something-Something v2 (동작 이해, top-1) | 77.3 |
| Epic-Kitchens-100 recall-at-5 (인간 행동 예측) | 39.7 (SOTA) |
| V-JEPA 2-AC 학습 데이터 | Droid 로봇 영상, 라벨 없이 62시간 미만 |
| 로봇 배치 | 서로 다른 두 실험실의 Franka 팔에 zero-shot |

로봇: **V-JEPA 2-AC를 서로 다른 두 실험실의 Franka 팔에 zero-shot 배치** — 해당 로봇에서 데이터 수집·태스크별 학습·보상 없이, **목표 이미지 기반 계획만으로** 픽앤플레이스 수행.

## 4. 스터디와의 개념적 연결 (가장 중요한 갈래)

- **우리 개인 방향과 정확히 일치**: "WM은 예측형(JEPA) 위주." 이 페이지가 트랙의 실질 앵커 후보다.
- **3D/기하 강점과 최근접**: 표현공간에서 물체·운동·기하 구조를 예측한다는 건, [Perception 파이프라인](../perception.md)에서 다룬 point cloud·pose·depth로 관측을 **기하 상태**로 요약하는 것과 같은 철학(태스크-무관 픽셀 버리기).
- **목표 이미지 기반 계획**은 이 사이트의 "인지 → 타깃 → 도달" 파이프라인과 직접 대응: V-JEPA 2-AC의 $\arg\min\|\hat s_{t+H}-s^\ast\|$는, pose 타깃으로 IK를 풀던 것을 **표현공간 MPC**로 일반화한 형태로 읽을 수 있다.
- **보상 없이 계획**: 보상 설계가 어려운 실조작에서, 목표 표현만으로 움직인다는 점이 WM-RL(Dreamer/MuZero)의 보상 의존 한계를 보완 — VLA(모방) vs WM-RL(보상) 비교는 [Concepts](../concepts.md) 참고.

## 5. 한 줄 평·한계

**한 줄 평.** "미래를 그리지 말고, 미래의 **표현**을 예측하라." [V-JEPA](vjepa.md)가 연 제3의 길을, 스케일과 행동 조건으로 **실로봇 계획**까지 끌고 간 게 결정적. 로봇 실용 최전선.

**한계.**
- **표현공간 목표의 정의**: "목표 표현 $s^\ast$"를 무엇으로/얼마나 정밀하게 줄지가 태스크 성패를 가른다. 정밀 6-DoF·접촉 풍부 조작에서 표현 예측만으로 충분한지는 열린 문제(**미확인** 영역).
- **평가 난이도**: 재구성이 없어 "무엇을 예측하는지"가 픽셀로 안 보인다(MuZero와 공유하는 해석성 문제).
- **장기·물리 일관성**: 표현공간 예측도 지평이 길면 흐려진다. 장기 계획은 여전히 짧은 지평 MPC로 상각.
- **로봇 검증 범위**: zero-shot Franka 픽앤플레이스는 인상적이나, 접촉 풍부·정밀 6-DoF 조작으로의 확장은 후속 과제(원문도 픽앤플레이스류 중심).
