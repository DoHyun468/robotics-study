# V-JEPA (2024)

작성: world-models 심화 트랙 · 원문 기준
- Bardes, Garrido, Assran 외 (Meta AI), *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA), arXiv 2404.08471, 2024.
- 계보: LeCun의 JEPA(Joint-Embedding Predictive Architecture) 구상 → I-JEPA(이미지) → **V-JEPA(영상)** → [V-JEPA 2](vjepa2.md)(스케일 + 로봇 계획).

> V-JEPA(v1) 수치는 원문(2404.08471) 대조 완료. 스케일업·행동 조건·로봇 계획으로 이어지는 후속작은 별도 페이지 [V-JEPA 2](vjepa2.md) 참고.

## 한 줄 요약

관측을 **재구성하지 않고**, 마스킹된 부분의 **표현(representation)**을 표현공간에서 예측하는 자기지도 world model. 픽셀 복원(Dreamer)도, 보상 예측(MuZero)도 아닌 **제3의 축(예측·표현형)**. 학습된 표현을 **frozen**으로 두고 downstream 벤치에서 평가해, 재구성·contrastive 없이도 강한 표현이 나옴을 보인 원조편.

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
- 행동 조건(action-conditioned) 확장이나 로봇 계획으로의 적용은 이 원조편에는 없음 — 그 부분은 [V-JEPA 2](vjepa2.md)에서 다룬다.

## 3. 결과

**V-JEPA (v1, 원문 보고)**: 재구성·contrastive 없이 **frozen** 표현만 평가. 최대 모델 **ViT-H/16(영상만 학습)** 기준:

| 벤치 | 지표 | 값 |
|---|---|---|
| Kinetics-400 | top-1 | 81.9 |
| Something-Something-v2 | top-1 | 72.2 |
| ImageNet-1K | top-1 | 77.9 |

영상 이해(운동·시간 구조)에서 특히 강함.

## 4. 스터디와의 개념적 연결

- **우리 개인 방향과 정확히 일치**: "WM은 예측형(JEPA) 위주." 이 갈래가 트랙의 실질 앵커 후보다.
- **3D/기하 강점과 최근접**: 표현공간에서 물체·운동·기하 구조를 예측한다는 건, [Perception 파이프라인](../perception.md)에서 다룬 point cloud·pose·depth로 관측을 **기하 상태**로 요약하는 것과 같은 철학(태스크-무관 픽셀 버리기). V-JEPA의 "예측 가능한 구조만 남긴다"는 depth/registration 관점의 학습판.
- **World model 3축** 중 예측·표현형의 원조 — 시퀀스·토큰형(Dreamer/MuZero), 생성·영상형과의 성숙도 비교는 [Concepts](../concepts.md) 참고.

## 5. 한 줄 평·한계

**한 줄 평.** "미래를 그리지 말고, 미래의 **표현**을 예측하라." 재구성의 낭비와 contrastive의 취약함을 동시에 피하는 제3의 길을, 로봇 계획 없이 **표현 품질만으로** 입증한 원조편.

**한계.**
- **frozen 평가의 한계**: probe로 측정한 표현 품질이 실제 행동·계획에 얼마나 전이되는지는 이 논문만으로는 확인 불가(**미확인** 영역) — 후속작 V-JEPA 2가 이 질문에 답한다.
- **행동 조건 없음**: 이 원조편은 action-conditioned 예측이나 실로봇 계획을 다루지 않는다.
- **평가 난이도**: 재구성이 없어 "무엇을 예측하는지"가 픽셀로 안 보인다(MuZero와 공유하는 해석성 문제).
