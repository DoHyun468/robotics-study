# 전신 자세 추정 — SMPL 멀티뷰 피팅 (실측)

[손(MANO) 멀티뷰 피팅](hand_pose.md)을 **관절 손 → 전신(SMPL body)** 으로 확장한 실측이다. 손과 완전히 동일한 방법론 — **GT를 알고, 관측에서 복원하고, 오차를 mm로 잰다** — 을 6890-버텍스·52-관절 전신 파라메트릭 모델에 적용한다. 여기서는 **SMPL forward를 라이브러리에 의존하지 않고 직접 구현**(shape/pose blendshape + Linear Blend Skinning)해, 각 수식이 무엇을 하는지 논문 수준으로 전개한다. 모델 리뷰는 [SMPL·MANO](reviews/smpl-mano.md).

## 1. SMPL 파라메트릭 모델 (원리)

SMPL은 사람 몸을 **저차원 파라미터**로 생성하는 미분가능 함수 $M(\beta,\theta)\to(V,J)$ 다. 입력은 **형상(shape)** $\beta\in\mathbb{R}^{10}$ 과 **자세(pose)** $\theta\in\mathbb{R}^{3(J+1)}$ (관절별 axis-angle, $J{=}$ 관절 수). 우리가 쓴 모델은 **SMPL+H**(SMPL 몸 + MANO 손, 52관절·6890버텍스)이며 HM0에서는 **몸 22관절**을 최적화하고 손은 flat mean으로 고정한다.

### 1.1 형상 blendshape — 체형
평균 체형(template) $\bar T\in\mathbb{R}^{6890\times3}$ 에 shape 주성분 $S_n$ 을 $\beta$ 만큼 더해 개인 체형을 만든다:

$$V_{\text{shaped}}=\bar T+\sum_{n=1}^{10}\beta_n\,S_n,\qquad S\in\mathbb{R}^{6890\times3\times10}.$$

- $\beta_n$: PCA 계수(키·체격 등). $S_n$: $n$번째 shape 방향(6890버텍스의 변위).

### 1.2 관절 회귀 — 뼈대 위치
체형이 정해지면 **관절 위치**는 버텍스의 선형결합으로 얻는다:

$$J=\mathcal{J}\,V_{\text{shaped}},\qquad \mathcal{J}\in\mathbb{R}^{52\times6890}.$$

$\mathcal{J}$(joint regressor)는 각 관절을 주변 표면 버텍스의 가중평균으로 정의한다.

### 1.3 자세 blendshape — LBS 아티팩트 보정
관절을 회전시키기 전에, **회전에 따른 표면 변형**(근육 벌크 등, 순수 스키닝으로는 안 나오는)을 보정한다. 비-루트 관절의 회전행렬에서 항등행렬을 뺀 값을 특징으로 쓴다:

$$V_{\text{posed}}=V_{\text{shaped}}+\sum_{k=1}^{J-1}\big(R(\theta_k)-I\big)\,P_k,\qquad P\in\mathbb{R}^{6890\times3\times 9(J-1)}.$$

- $R(\theta_k)\in SO(3)$: 관절 $k$ 의 회전(§1.4). $(R-I)$ 를 씀으로써 **rest pose($\theta{=}0$)에서 보정=0**.

### 1.4 축각 → 회전 (Rodrigues)
각 관절 자세는 axis-angle $\theta_k=\alpha\,\hat n$ ($\alpha=\|\theta_k\|$):

$$R(\theta_k)=I+\frac{\sin\alpha}{\alpha}[\theta_k]_\times+\frac{1-\cos\alpha}{\alpha^2}[\theta_k]_\times^2.$$

(SLAM의 SE(3)에서 쓴 것과 같은 지수사상.)

### 1.5 Linear Blend Skinning (LBS) — 뼈대에 살 붙이기
관절을 **kinematic tree**(부모→자식)로 누적해 각 관절의 전역 변환 $G_k$ 를 만들고, 각 버텍스를 인접 관절들의 변환으로 가중혼합한다:

$$G_k=\prod_{a\in\text{ancestors}(k)}\begin{bmatrix}R(\theta_a)&J_a-J_{\text{parent}(a)}\\\mathbf0&1\end{bmatrix},\qquad
v_i'=\sum_{k=1}^{J} w_{ki}\,G'_k\begin{bmatrix}v_{i,\text{posed}}\\1\end{bmatrix}.$$

- $w_{ki}$: 스키닝 가중치(버텍스 $i$ 가 관절 $k$ 에 얼마나 묶이는가, $\sum_k w_{ki}=1$).
- $G'_k=G_k\cdot\mathrm{rest\text{-}removal}$: rest pose를 항등으로 만드는 보정 $G'_k=G_k-[\,\mathbf0\ |\ G_k(J_k;0)\,]$ (rest에서 $v'=v$ 보장).
- 최종 관절 위치는 $\hat J_k=G_k[:3,3]$.

> **한 줄 요약:** $\beta$ 로 체형을 만들고 → 관절 위치를 회귀하고 → 자세 보정을 더하고 → 뼈대 회전을 kinematic tree로 누적해 → 가중 스키닝으로 표면을 움직인다. 전부 미분가능 → 관측에서 $(\beta,\theta)$ 를 **역으로 최적화**할 수 있다.

## 2. 우리 실측 — 멀티뷰 복원 피팅

1. **GT 몸 샘플**: $\beta\sim\mathcal N(0,1)$, 몸 관절 자세 $\theta$ 를 무작위(±0.25 rad)로 뽑아 SMPL forward → GT 3D 관절·메시.
2. **N개 캘리브 카메라**(링 배치, 고정 $K$·extrinsic)에 몸 관절을 투영 → 2D + **픽셀 노이즈** 주입.
3. **복원**: $(\beta,\theta,\text{transl})$ 을 0에서 시작해 멀티뷰 재투영오차를 Adam으로 최소화:

$$\min_{\beta,\theta,\mathbf t}\ \sum_{c=1}^{N}\sum_{j}\rho\!\big(\|\pi_c(\hat J_j(\beta,\theta)+\mathbf t)-x_{cj}\|\big)+\lambda_\beta\|\beta\|^2+\lambda_\theta\|\theta_{\text{body}}\|^2,$$

$\pi_c$ = 카메라 $c$ 의 핀홀 투영, $\rho$ = pseudo-Huber(이상치 완화), 정규화 $\lambda$ 항은 shape/pose prior.
4. GT 대비 **MPJPE(mm)**·버텍스오차·$\beta$ 오차를 3-seed 평균으로 측정.

<img src="_static/hm_smpl_fit.png" alt="SMPL multi-view body fitting" style="width:100%;max-width:1000px;border-radius:8px">

*좌: GT 관절(teal)과 복원(주황 ×)·뼈대가 3D에서 겹침. 중: cam-1 재투영(회색=복원 메시 6890버텍스). 우: shape $\beta$ GT vs 복원.*

## 3. 결과 (3-seed 평균)

**① 뷰 수** (노이즈 1px)

| 뷰 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| MPJPE | **157.3mm** | 9.3mm | 9.0mm | 8.5mm |

**단일 뷰는 157mm로 붕괴**한다 — 전신은 metric depth/scale 모호성이 손보다 훨씬 심하다(팔·다리 전후 위치가 한 뷰로 결정 안 됨). **2뷰만 돼도 9.3mm로 −94%**, 그 뒤 포화. 손에서 본 "멀티뷰 기하가 단안의 약제약을 푼다"가 전신에서 **훨씬 극적**으로 재현된다.

**② 픽셀 노이즈** (4뷰)

| 노이즈 | 0px | 1px | 2px | 4px |
|---|---|---|---|---|
| MPJPE | 9.0mm | 9.0mm | 8.7mm | 11.2mm |

손(노이즈에 거의 선형 전파)과 달리 **전신 4뷰 피팅은 2px까지 노이즈에 강건**하고 4px에서만 오른다. ~9mm 바닥은 픽셀노이즈가 아니라 **shape–pose 결합/최적화 수렴**이 지배한다(아래 β).

**③ shape $\beta$ 는 키포인트로 완전히는 안 잡힌다**: $\beta$ 오차 ~1.3로 남고 버텍스오차(~18mm)가 관절오차(~9mm)보다 크다. **키포인트는 pose를 제약하지 shape(체형)를 제약하지 않는다** — 손 [E4](hand_pose.md)에서 표면 관측을 더해 풀었던 것과 동일한 한계. 전신 체형 복원엔 실루엣/표면 항이 필요하다.

## 4. 정직/한계
- SMPL forward를 직접 구현(shape/pose blendshape + LBS)했고, GT 생성·복원에 **동일 forward**를 써 자기일관적으로 fitting 성능만 측정한다(모델 자체의 캐노니컬 정확도가 아니라 **복원 오차** 연구).
- 몸 22관절만 최적화, 손·얼굴은 flat 고정(전신+손+얼굴 통합은 HM1: SMPL-X).
- 시뮬 GT 벤치. 실데이터(monocular HMR2.0 등)는 HM6.
- ~9mm 바닥은 from-scratch 일반 옵티마이저의 수렴/β 결합 한계 — 회귀 초기화나 표면항으로 개선 여지.

## 5. 연결
[손(MANO) 관절체 피팅](hand_pose.md)이 "변형 가능한 형상 모델을 알면 손 관절이 잡힌다"였다면, 이건 그 구조가 **6890버텍스·52관절 전신**에서 동일하게(멀티뷰·노이즈·shape 관측성) 성립함을 보인다. 다음: **SMPL-X 전신+손+얼굴**(HM1), **HOI 손-물체 공동**(HM2), **egocentric**(HM3), **다중센서 동기화**(HM4), **Human→Robot retargeting**(HM5). 전체 계획은 `robotics-lab/notes/human_plan.md`.

## 재현
```bash
# SMPL+H 모델(license 등록 다운로드)을 <dir>/smplh/SMPLH_MALE.pkl에 두고:
python src/hm_smpl_fit.py --model <dir>/smplh/SMPLH_MALE.pkl --sweep --render
```
