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

### 무엇을 입력으로 보나 (how it works)

그래프만으론 뭘 하는지 안 보이니, **방법이 실제로 보는 입력**부터 보인다. 아래는 SMPL 몸을 4개 캘리브 카메라에서 렌더한 것(음영 메시)이다 — 이게 "관측"이고, 최적화는 이 이미지들의 **2D 관절**(teal)에 SMPL 재투영(orange ×)이 겹치도록 $(\beta,\theta)$ 를 움직인다.

<img src="_static/hm_smpl_views.png" alt="what each camera sees: shaded SMPL body + 2D joints" style="width:100%;max-width:1000px;border-radius:8px">

*카메라 1–4가 같은 사람을 다른 각도에서 본다(음영=SMPL 메시 6890버텍스). teal=관측 2D 관절, orange ×=현재 추정의 재투영. 한 장(단안)으론 전후 depth가 모호 → 여러 각도가 그 모호성을 푼다.*

> **파이프라인 한눈에:** ① N개 캘리브 카메라가 몸을 관측 → ② 각 뷰에서 2D 관절 검출(여기선 GT 투영+노이즈로 대체) → ③ SMPL $(\beta,\theta,\mathbf t)$ 를 **모든 뷰의 재투영오차 최소화**로 최적화(§1의 forward가 미분가능하므로 gradient descent) → ④ 3D 몸(관절·메시) 복원.

<img src="_static/hm_smpl_fit.png" alt="SMPL multi-view body fitting result" style="width:100%;max-width:1000px;border-radius:8px">

*결과: 좌=GT 관절(teal)과 복원(주황 ×)·뼈대가 3D에서 겹침, 중=cam-1 재투영(회색=복원 메시), 우=shape $\beta$ GT vs 복원.*

## 3. 결과 (3-seed 평균)

**① 뷰 수** (노이즈 1px)

| 뷰 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| MPJPE | **119.1mm** | 9.5mm | 8.5mm | 7.6mm |

**단일 뷰는 119mm로 붕괴**한다 — 전신은 metric depth/scale 모호성이 손보다 훨씬 심하다(팔·다리 전후 위치가 한 뷰로 결정 안 됨). **2뷰만 돼도 9.5mm로 −92%**, 그 뒤 완만히 개선(8뷰 7.6mm). 손에서 본 "멀티뷰 기하가 단안의 약제약을 푼다"가 전신에서 **훨씬 극적**으로 재현된다.

**② 픽셀 노이즈** (4뷰)

| 노이즈 | 0px | 1px | 2px | 4px |
|---|---|---|---|---|
| MPJPE | 7.8mm | 8.5mm | 9.4mm | 9.9mm |

2D 검출 노이즈가 3D 오차로 **단조 전파**(7.8→9.9mm) — [perception 오차예산](perception.md)·[손 E1](hand_pose.md)의 "입력 오차→출력 오차 전파"가 전신에서도 성립.

**③ shape $\beta$ 는 키포인트로 완전히는 안 잡힌다**: $\beta$ 오차 ~1.7로 남고 버텍스오차(~18mm)가 관절오차(~8mm)보다 크다. **키포인트는 pose를 제약하지 shape(체형)를 제약하지 않는다** — 손 [E4](hand_pose.md)에서 표면 관측을 더해 풀었던 것과 동일한 한계. 전신 체형 복원엔 실루엣/표면 항이 필요하다.

## 4. 정직/한계
- SMPL forward를 직접 구현(shape/pose blendshape + LBS)했고, GT 생성·복원에 **동일 forward**를 써 자기일관적으로 fitting 성능만 측정한다(모델 자체의 캐노니컬 정확도가 아니라 **복원 오차** 연구).
- 몸 22관절만 최적화, 손·얼굴은 flat 고정(전신+손+얼굴 통합은 HM1: SMPL-X).
- 시뮬 GT 벤치. 실데이터(monocular HMR2.0 등)는 HM6.
- ~9mm 바닥은 from-scratch 일반 옵티마이저의 수렴/β 결합 한계 — 회귀 초기화나 표면항으로 개선 여지.

## 5. 연결
[손(MANO) 관절체 피팅](hand_pose.md)이 "변형 가능한 형상 모델을 알면 손 관절이 잡힌다"였다면, 이건 그 구조가 **6890버텍스·52관절 전신**에서 동일하게(멀티뷰·노이즈·shape 관측성) 성립함을 보인다. 다음: **SMPL-X 전신+손+얼굴**(HM1), **HOI 손-물체 공동**(HM2), **egocentric**(HM3), **다중센서 동기화**(HM4), **Human→Robot retargeting**(HM5). 전체 계획은 `robotics-lab/notes/human_plan.md`.

## 6. HM1 — whole-body(몸 + 양손) + β/scale 추정

HM0(몸 22관절)에서 **양손 30관절까지 함께** 최적화하고, JD가 콕 집은 **global scale** 파라미터를 추가해 "shape/pose 최적화(β/scale 추정)"를 정면으로 다룬다. (얼굴까지는 SMPL-X 모델이 필요 — 여기선 SMPL+H로 몸+손.)

**입력(무엇을 보나):** 몸 카메라가 보는 것 — 손은 프레임에서 아주 작다(수십 px).

<img src="_static/hm1_views.png" alt="whole-body input views" style="width:100%;max-width:1000px;border-radius:8px">

*4개 카메라에서 본 사람(음영 메시) + 52개 관절(teal 관측 / orange × 재투영). 손 관절이 몸 카메라에선 작게 뭉쳐 있다 → 약하게 제약됨.*

<img src="_static/hm1_wholebody.png" alt="whole-body fit + beta/scale" style="width:100%;max-width:1000px;border-radius:8px">

*좌=whole-body GT vs 복원(몸+손), 중=shape β, 우=global scale 복원.*

**① scale 관측성 (β/scale 추정의 핵심)**

| 뷰 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| scale 오차 | **4.3%** | 1.1% | **0.8%** | 1.1% |
| body MPJPE | 247.9mm | 8.3mm | 8.0mm | 7.9mm |

**단안은 scale-depth 모호성**으로 크기를 못 잡는다(4.3%, 몸도 248mm로 붕괴) — 큰 사람이 멀리 있는 것과 작은 사람이 가까이 있는 게 한 뷰에선 동일하게 보이기 때문. **metric 멀티뷰 리그는 baseline이 절대 크기를 고정**해 scale을 <1%로 복원한다. 실무에서 "카메라 리그로 metric scale 확보"가 왜 중요한지의 정량 근거.

**② 손은 몸 카메라에서 훨씬 어렵다**

| 뷰 | 2 | 4 | 8 |
|---|---|---|---|
| body MPJPE | 8.3mm | 8.0mm | 7.9mm |
| **hand MPJPE** | 20.8mm | 19.4mm | **17.2mm** |

**손 오차가 몸의 ~2.4배**(4뷰 19 vs 8mm). 손이 몸 카메라 프레임에서 수십 px밖에 안 돼 관절이 약하게 제약되기 때문 — **정밀한 손 자세엔 근접/egocentric(머리장착) 카메라가 필요**하다는 정량 근거(→ HM3). shape β도 whole-body에선 더 어렵다(β 오차 ~1.9).

## 7. HM2 — 손-물체 상호작용(HOI) + 접촉 prior

손(MANO)이 물체(box)를 잡는 장면에서 **손 파라미터와 물체 6-DoF를 동시에** 복원한다. HOI의 핵심 재료는 **접촉/침투(contact/penetration) prior** — 손이 물체를 뚫고 들어가면 안 되고(물리적 타당성), 접촉해야 한다.

**입력(무엇을 보나):** 손(MANO 메시)과 물체(box)를 4개 카메라에서 본 것.

<img src="_static/hm2_views.png" alt="HOI input: hand+object mesh per camera" style="width:100%;max-width:1100px;border-radius:8px">

*카메라별 손+물체 메시(어두운 부분=box). teal=관측 2D 관절(손 21 + box 8코너), orange ×=재투영.*

**방법:** 손(β,pose,global,transl) + 물체(6-DoF)를 함께 최적화, 재투영오차 + **침투 penalty**. box는 해석적 SDF를 쓴다: 점 $p$(물체 프레임), 반경 $h$에 대해 $q=|p|-h$, $\text{sd}(p)=\|\max(q,0)\|+\min(\max_i q_i,0)$ (안=음수). 손 버텍스의 침투 $\sum_v \max(0,-\text{sd}(v_o))^2$ 를 penalize.

| | 손 MPJPE | 침투(penetration) |
|---|---|---|
| **1-view, 접촉 prior 없음** | 14.2mm | **9.1mm** (손이 물체를 뚫음) |
| **1-view, +접촉 prior** | **6.4mm** (−55%) | **0.0mm** |
| 4-view, 접촉 없음 | 1.1mm | 0.0mm |
| 4-view, +접촉 | 1.1mm | 0.0mm |

**핵심(HOI 접촉 prior의 값):** 단안(1-view)에선 손-물체 상대 depth가 모호해 **손이 물체를 9.1mm 뚫고 들어간다**(재투영은 맞지만 물리적으로 틀림). **접촉 prior가 침투를 0으로 없애고 손 오차를 절반(14→6.4mm)으로** 줄인다 — 물리적으로 타당한 재구성. **멀티뷰(4)는 기하만으로 이미 해결**(손 1.1mm·물체 0.9mm·침투 0)이라 접촉항이 비활성. HOLD·GRAB·ARCTIC 계열이 접촉 prior를 쓰는 이유를 그대로 보인다.

*정직: 시뮬 clean 멀티뷰에선 접촉항이 불필요(재투영으로 충분). 접촉 prior의 결정적 값은 **실제 노이즈 있는 단안/가림** 상황(HM6)이며, 여기선 그 메커니즘을 단안 depth 모호성으로 재현했다.*

## 8. HM3 — Egocentric vs Exocentric: 손엔 근접 카메라가 정공법

HM1·HM2에서 **손은 몸/원거리(exocentric) 카메라에서 약하게 제약**됨을 봤다(수십 px). HM3는 그 해법을 정량화한다: **머리장착(egocentric) 근접 카메라**는 손이 프레임을 가득 채워, 같은 2D 검출 노이즈가 훨씬 작은 3D 오차로 매핑된다.

<img src="_static/hm3_ego_vs_exo.png" alt="egocentric vs exocentric hand view" style="width:100%;max-width:1100px;border-radius:8px">

*같은 해상도·같은 손인데 — 원거리 exo(좌)에선 손이 45px 티끌, 근접 ego(우)에선 327px로 프레임을 채운다(7×).*

| 카메라 구성 | 손 MPJPE | 비고 |
|---|---|---|
| 1× exo (far ~1.8m) | 111.2mm | 손 45px, 단안 |
| 1× ego (close ~0.3m) | 72.2mm | 손 327px — 근접이 단안 exo보다 나음 |
| **2× ego (stereo 7cm)** | **6.2mm** | 근접 스테레오 → metric depth (Quest/Aria류) |
| 4× exo (far) | 2.1mm | 원거리라도 멀티뷰면 해결 |
| ego + 3× exo | 3.0mm | 융합 |
| 1× ego, **가림 40%** | 77.3mm | 단일뷰는 자기가림에 취약 |
| **ego + 3× exo, 가림 40%** | **3.9mm** | 융합이 가림 보완 |

**핵심:**
1. **근접(ego)이 원거리(exo) 단안보다 낫다**(72 vs 111mm) — 손이 7× 크게 잡혀 2D가 정밀.
2. **단일 뷰는 어떤 것이든 depth 모호**(72–111mm) → mm 정확도엔 멀티카메라 필요.
3. **Ego 스테레오가 metric depth를 준다**(6.2mm) — 실제 헤드셋(Quest·Aria·[HOT3D](https://facebookresearch.github.io/hot3d/))이 근접 스테레오 카메라를 쓰는 이유.
4. **ego+exo 융합은 가림에 강건**(가림 40%에서 단일 ego 77mm → ego+3exo 3.9mm) — 한 뷰에서 가려진 손가락을 다른 뷰가 채운다.

## 9. HM4 — 다중센서 시공간 캘리브레이션·동기화 (camera↔IMU)

perception·hand-eye에서 한 건 **spatial** 캘리브였다. HM4는 빠져 있던 **temporal(시간 오프셋 $t_d$) + IMU**를 채운다. 강체로 붙은 두 센서가 같은 모션을 보지만 (a) **클럭이 $t_d$만큼 어긋나고** (b) 프레임이 **extrinsic 회전 $R_{ci}$** 로 다르다. 둘을 **각속도 스트림 정합**으로 동시 추정한다 — Kalibr/VINS online temporal calibration의 핵심 원리.

$$w_\text{imu}(t)=R_{ci}^\top\,w_\text{cam}(t-t_d)+\text{noise};\qquad \min_{t_d,\,R_{ci}}\ \sum_t\big\|\,w_\text{imu}(t)-R_{ci}^\top\,\text{interp}\big(w_\text{cam},\,t-t_d\big)\big\|^2$$

$t_d$에 대한 **미분가능 선형보간**으로 gradient descent가 동기화를 푼다.

<img src="_static/hm4_sync.png" alt="camera-IMU temporal+spatial calibration before/after" style="width:100%;max-width:1100px;border-radius:8px">

*위=보정 전(카메라 solid vs IMU dashed): $t_d$=60ms 어긋남 + extrinsic 34°로 축이 섞임. 아래=추정 후($t_d$·$R_{ci}$ 복원): 두 스트림이 겹친다.*

| 조건 | $t_d$ 오차 | extrinsic 오차 |
|---|---|---|
| baseline (노이즈 0.02 rad/s, $t_d$ 30ms) | **2.1 ms** | **0.26°** |
| 노이즈 0 / 0.05 / 0.1 | 0 / 8.5 / 75 ms | 0 / 1.1 / 7.0° |
| **모션 amp 0.15 / 0.4 / 1.0** | 54.8 / 9.6 / **2.1 ms** | 4.9 / 1.2 / 0.26° |

**핵심:** 시간 오프셋을 **~2ms**, extrinsic 회전을 **0.26°** 로 복원. 노이즈에 단조 열화. **모션 excitation이 관측성의 열쇠** — 회전이 작으면(amp 0.15) 55ms/4.9°로 무너지고, 충분히 흔들면(amp 1.0) 2ms/0.26°. Kalibr가 "모든 축을 충분히 흔들라"고 요구하는 이유를 실측으로 확인. *정직: 각속도(gyro)로 $t_d$·회전 extrinsic까지. 완전한 spatial(translation lever-arm)엔 accelerometer가 추가로 필요(확장 가능).*

## 10. HM5 — 사람 손 → 로봇 핸드 dexterous retargeting

여기까지(HM0–HM4)는 **사람을 관측·복원**하는 perception이었다. HM5는 그 결과를 **로봇이 실행할 관절 명령으로 변환**한다 — JD의 "Human-to-Robot Motion Retargeting / 구조적 차이를 고려한 grasp pose 매핑 / 물리적 실현 가능성 검증 및 후처리(smoothing·interpolation·collision avoidance)". 대상 로봇은 **Allegro 4손가락 16-DoF 핸드**(MuJoCo Menagerie `wonik_allegro`), 소스는 **MANO 사람 손**(5손가락, ~21 DoF, 연조직).

### 10.1 왜 관절각을 그대로 복사하면 안 되는가

사람 손과 로봇 핸드는 **kinematic 구조가 다르다** — 손가락 수(5 vs 4), 링크 길이, 관절축, 가동범위, 그리고 사람의 연조직 변형. 사람 관절각 $\theta_\text{human}$ 을 로봇에 그대로 넣으면 **작업 공간(fingertip 위치)이 어긋난다**. 그래서 관절각이 아니라 **task-space 기하량(keyvector)** 을 맞춘다 — DexPilot / AnyTeleop(dex-retargeting)의 원리.

### 10.2 Keyvector 정의

손바닥(palm) 원점에서 각 손가락의 **중간 마디(medial)와 끝(distal)** 으로 향하는 벡터를 정의한다. 손가락 4개(검지 ff · 중지 mf · 약지 rf · 엄지 th) × 2마디 = **8개 keyvector**:

$$v^{(h)}_i = p^{(h)}_i - p^{(h)}_\text{palm}\ \ (\text{사람}),\qquad v^{(r)}_i(q) = p^{(r)}_i(q) - p^{(r)}_\text{palm}(q)\ \ (\text{로봇, } q\text{의 함수})$$

로봇 쪽 $p^{(r)}_i(q)$ 는 MuJoCo forward kinematics(`mj_forward`)로 얻는다. **중간+끝 두 마디**를 함께 잡는 게 중요하다 — 끝점만 맞추면 손가락이 "닿기만" 하고 **굽힘(curl)** 이 제약되지 않아 로봇이 덜 오므린다(실제로 처음 tip만 썼을 때 fist에서 안 접히는 버그를 겪음).

### 10.3 프레임 정렬과 스케일

사람 손과 로봇 손은 크기·기준 방향이 다르다. **스케일** 은 open 자세의 평균 keyvector 길이 비로,

$$s = \frac{\frac{1}{8}\sum_i \|v^{(r)}_i(0)\|}{\frac{1}{8}\sum_i \|v^{(h)}_i\|}\ (\approx 0.93)$$

**회전 정렬** $R$ 은 open 자세에서 두 keyvector 집합을 맞추는 **Kabsch(직교 Procrustes)** 해 — $H=\sum_i (s\,v^{(h)}_i)(v^{(r)}_i)^\top=U\Sigma V^\top$, $R=V\,\mathrm{diag}(1,1,\det(VU^\top))\,U^\top$ (반사 방지). 목표 keyvector는 $\;\tilde v_i = s\,R\,v^{(h)}_i$.

### 10.4 경계 최적화 = 실현 가능성(feasibility) 내장

로봇 관절각 $q\in\mathbb{R}^{16}$ 을 아래로 푼다 — **관절 한계를 bound로 넣어** 해가 항상 물리적으로 실행 가능:

$$\min_{q}\ \sum_{i=1}^{8}\big\|\,v^{(r)}_i(q)-\tilde v_i\,\big\|^2 \;+\; \lambda_p\big\|(v^{(r)}_{th}-v^{(r)}_{ff})-(\tilde v_{th}-\tilde v_{ff})\big\|^2 \quad \text{s.t.}\quad q_\text{lo}\le q\le q_\text{hi}$$

둘째 항은 **엄지–검지 pinch**(집기)를 강조하는 상대벡터 항($\lambda_p=1.5$). **L-BFGS-B**(bound-constrained)로 풀며, 시간 축에서는 이전 프레임 해로 **warm-start** 해 궤적 일관성을 준다. bound 덕분에 **joints within limits = 100%** 가 설계상 보장된다.

### 10.5 Self-collision 회피 (후처리 ①)

사람이 주먹을 꽉 쥐면 로봇 손가락끼리 **파고든다(interpenetration)**. 순수 retargeting은 24프레임 중 **6프레임이 자기충돌**했다. MuJoCo 접촉으로 최대 침투깊이 $d_\text{pen}(q)=\max(0,\,-\min_c \text{dist}_c)$ 를 읽어 **패널티**로 넣는다:

$$\min_q\ \big[\text{(위 식)}\big] + w_\text{col}\,d_\text{pen}(q)^2,\qquad w_\text{col}=400$$

결과: 자기충돌 **6/24 → 0/24**, 그러면서 keyvector 정확도는 **58.6 → 58.7mm** 로 거의 그대로(0.1mm 손해) — 충돌 회피가 사실상 공짜.

### 10.6 Smoothing·Interpolation (후처리 ②③)

- **Smoothing:** 관절 궤적에 폭 5 이동평균(경계는 edge-pad로 ringing 방지). jerk $\frac{1}{T}\sum_t|\,q_{t+1}-3q_t+3q_{t-1}-q_{t-2}|$ 가 **0.045 → 0.009 (5×↓)** — 실행 시 급가속 억제.
- **Interpolation:** 24프레임을 **2× 업샘플(48)** 선형보간해 매끄러운 실행 궤적으로. 보간 후에도 자기충돌 **0/48** 유지.

<img src="_static/hm5_retarget.png" alt="MANO human grasp retargeted to Allegro dexterous hand" style="width:100%;max-width:1200px;border-radius:8px">

*위=사람 MANO 손(open→쥠→fist). 아래=리타게팅된 Allegro 핸드 — 손가락이 사람을 따라 오므라든다. 사람 5손가락이 로봇 4손가락으로 매핑되며(약지↔로봇 약지, 새끼는 생략), 엄지 대향(opposition)이 로봇 엄지로 근사된다.*

| 항목 | 값 |
|---|---|
| human→robot 스케일 $s$ | 0.93 |
| keyvector 매칭 (raw → collision-safe) | 58.6 → **58.7 mm** |
| joints within limits | **100%** (bound 내장) |
| self-collision 프레임 | 6/24 → **0/24** → 0/48(보간 후) |
| jerk (smoothing) | 0.045 → **0.009** |

**핵심:** 8개 keyvector 매칭 + Kabsch 정렬 + bound 최적화로 **관절 한계·자기충돌을 모두 만족하는 실행 가능한 grasp 궤적**을 만들고, smoothing·interpolation까지 붙였다. *정직: 잔차 ~58mm는 **cross-morphology 구조 차이**(사람 5손가락↔로봇 4손가락, 엄지 대향 구조가 특히 다름)에서 오는 근본 한계로, 완벽히 0이 될 수 없다 — 그래서 절대 위치가 아닌 keyvector(상대 기하)를 맞춘다. 실로봇 배포엔 (a) 팔 IK를 붙여 손목 6-DoF 궤적까지, (b) 접촉면 mesh 기반 충돌, (c) 힘/토크 feasibility를 추가한다(확장 가능).*

## 11. HM6 — 실데이터 검증 (실제 사진 → 우리 파라메트릭 모델 피팅)

HM0–HM5는 전부 **시뮬레이션**이었다(GT를 알기에 mm 오차를 잴 수 있었음). HM6는 그 파이프라인을 **진짜 픽셀**에 돌려 닫는다. 레시피는 손·전신 공통이다: 실제 사진에서 **MediaPipe**로 2D 랜드마크를 검출하고(검출은 남의 모델), 그 검출에 **우리가 직접 구현한 파라메트릭 모델**(손=MANO, 전신=SMPL)을 약투영 단안 최적화로 맞춘다(피팅은 우리 코드). GT 3D가 없으니(실사진 1장) 품질 지표는 **2D 재투영 오차(px)** 이고, 이는 HM0–HM3의 단안 모호성 교훈을 실데이터에서 재확인하는 **정성 검증**이다.

### 11.1 손 (MANO)

실제 손 사진에서 **MediaPipe Hands**로 21점을 검출하고 우리 MANO를 피팅한다.

**방법(약투영·weak-perspective 단안 피팅):** MediaPipe 21점을 MANO 16관절(손목 + 손가락별 MCP/PIP/DIP)에 대응시키고, MANO 자세 $\theta$(PCA)와 약투영 카메라(스케일 $s$·평행이동 $t$)를 아래로 최적화한다 — 스케일-깊이 모호를 피하려 원근 대신 약투영을 쓴다:

$$\min_{\theta,\,s,\,t}\ \sum_{j=1}^{16}\Big\|\,s\,m\,P_{xy}\big(\hat J_j(\theta)\big)+t\;-\;x^{\text{det}}_j\,\Big\|\;+\;\lambda\|\theta\|^2$$

$P_{xy}$ = 관절의 $(x,y)$ 성분(직교투영), $m{=}{\pm}1$ 좌우손 반전, $x^{\text{det}}_j$ = MediaPipe 검출 2D. 2단계(카메라+전역회전 먼저 → 관절 자세 추가) Adam.

<img src="_static/hm6_realfit.png" alt="real photos -> MediaPipe 2D -> our MANO fit" style="width:100%;max-width:1200px;border-radius:8px">

*위=실제 사진 + MediaPipe 검출(green) vs 우리 MANO 재투영(orange ×). 아래=복원된 3D MANO를 **40° 회전**해 본 것 — pointing은 검지, victory는 두 손가락이 실제로 3D에서 펴져 복원됨(단안 사진 1장에서).*

| 실제 사진 (제스처) | 손 | 2D 재투영 오차 |
|---|---|---|
| pointing up | Left | **8.8 px** (1.5% diag) |
| thumbs up | Right | 10.8 px |
| victory | Right | 16.5 px |
| woman hands | Right | 27.3 px |
| woman hands | Left | 40.4 px |
| **평균 (5 hands)** | | **20.8 px** (2.4% diag) |

**핵심:** 시뮬로 만든 우리 MANO 피팅이 **실제 사진에서도 작동** — 명확한 제스처는 2D 재투영 8–17px, 3D 자세(pointing/victory)가 의미적으로 옳게 복원된다. *정직: (1) **단안이라 3D GT가 없어 정성**이다(mm 못 잼) — 2D는 잘 맞아도 out-of-plane 깊이는 모호(HM0–HM3에서 정량화한 그 모호성). (2) `woman hands`가 27–40px로 더 나쁜 건 손이 기울어 foreshortening + 팔에 일부 가림 때문. (3) MediaPipe(검출)는 남의 모델이고, **피팅은 우리 코드**다.*

### 11.2 전신 (SMPL)

같은 레시피를 전신에 적용한다: **MediaPipe Pose**로 33점 body 랜드마크를 검출하고, **HM0에서 직접 구현한 SMPL**(shape/pose blendshape + LBS)을 피팅한다 — 관절이 63-DoF로 많고 2D 대응이 14점뿐이라 body 자세에 prior를 걸고, 관절별 **가시성(visibility)으로 가중**한다.

$$\min_{\theta,\,s,\,t}\ \sum_{j} w_j\Big\|\,s\,P_{xy}\big(\hat J_j(\theta)\big)+t\;-\;x^{\text{det}}_j\,\Big\|\;+\;\lambda\|\theta_{\text{body}}\|^2,\qquad w_j=\text{visibility}_j$$

대응: MediaPipe {어깨·팔꿈치·손목·엉덩이·무릎·발목·코} → SMPL {16/17·18/19·20/21·1/2·4/5·7/8·15}, 골반(SMPL 0)은 좌우 엉덩이 중점으로 합성.

<img src="_static/hm6_bodyfit.png" alt="real photos -> MediaPipe Pose 2D -> our SMPL fit" style="width:100%;max-width:1100px;border-radius:8px">

*위=실제 사진 + MediaPipe Pose(green) vs 우리 SMPL 재투영(orange ×). 아래=복원된 3D SMPL을 30° 회전 — 요가 warrior 자세(팔 벌림 + 런지)가 단안 사진 1장에서 3D로 복원된다.*

| 실제 사진 | 2D 재투영 오차 |
|---|---|
| yoga warrior (`pose`) | **1.9 px** (0.16% diag) |
| running (`bolt`, 동적) | 55.0 px (6.7% diag) |

**핵심:** 우리 SMPL 피팅이 실사진에서 작동 — 정적·명확한 자세(yoga)는 **재투영 1.9px**로 3D 전신을 정확히 복원한다. *정직: `bolt`가 55px로 나쁜 건 (a) 질주 중 다리가 강하게 foreshortening되고 (b) 단안이라 out-of-plane 다리 각도가 모호 + (c) 14점·prior 정규화로 극단 자세가 눌리기 때문 — HM0/HM1에서 본 단안 붕괴가 실데이터에서도 그대로.*

### 11.3 최적화(우리) vs 학습 회귀(HMR2.0) 대조

같은 실사진 2장에 **SOTA 학습 회귀 모델 [HMR2.0](https://shubham-goel.github.io/4dhumans/)**(4D-Humans, ViT-Huge 백본이 픽셀에서 SMPL 파라미터를 **직접 회귀**)을 돌려 우리 최적화 피팅과 대조했다. 우리 방식은 검출된 2D 키포인트에 **재투영을 직접 최소화**하고, HMR2.0은 2D를 **보지 않고** 이미지에서 학습된 prior로 3D를 추정한다는 게 근본 차이다. (SMPL neutral 모델 라이선스 필요, CPU 추론.)

<img src="_static/hmr2_realfit.png" alt="HMR2.0 learned SMPL regression on the same real photos" style="width:100%;max-width:1100px;border-radius:8px">

*HMR2.0 결과: 위=메시 투영 오버레이(green=MediaPipe, orange ×=HMR2.0 재투영), 아래=복원 3D 30° 회전. bolt의 질주 lean, pose의 warrior 자세가 3D로 복원된다.*

| 실제 사진 | 우리 SMPL (최적화) | HMR2.0 (학습 회귀) |
|---|---|---|
| yoga warrior (정적·명확) | **1.9 px** | 9.6 px |
| running bolt (동적·foreshortening) | 55.0 px | **18.1 px** |

**핵심 통찰 (둘의 성격이 정반대):**
- **정적·명확한 자세**에선 **최적화(우리)가 2D 재투영에서 더 낮다**(1.9 vs 9.6px) — 그 키포인트에 직접 맞추니까. 단, 이건 *2D* 지표이고 HMR2.0의 3D가 더 정확할 수 있다(2D를 과적합하지 않으므로).
- **동적·모호한 자세**에선 **학습 회귀(HMR2.0)가 압도적**이다(18 vs 55px). 단안 depth/foreshortening 모호를 **데이터에서 배운 prior**로 풀기 때문 — 우리 단일이미지 최적화는 그 모호에서 붕괴(bolt 55px)했지만 HMR2.0은 그럴듯한 질주 3D를 복원.
- 이게 이 분야의 교과서적 교훈이다: **최적화 = 정확하지만 취약(brittle), 학습 회귀 = 강건하지만 데이터 의존.** 그래서 실제 SOTA 파이프라인은 **회귀로 초기화 → 최적화로 정밀화**(SMPLify/HMR2+fitting)로 둘을 결합한다. 우리 트랙은 최적화 쪽(HM0–HM6)을 직접 구현했고, 여기서 회귀 SOTA와 정량 대조까지 했다.

*정직: HMR2.0은 남의 사전학습 모델이고 neutral SMPL(라이선스)이 필요하다. mm 단위 3D GT 벤치는 여전히 [HOT3D](https://facebookresearch.github.io/hot3d/)·3DPW/EMDB 같은 라이선스 데이터셋이 있어야 한다.*

## 재현
```bash
# SMPL+H 모델(license 등록 다운로드)을 <dir>/smplh/SMPLH_MALE.pkl에 두고:
python src/hm_smpl_fit.py   --model <dir>/smplh/SMPLH_MALE.pkl --sweep --render   # HM0 body
python src/hm1_wholebody.py --model <dir>/smplh/SMPLH_MALE.pkl --sweep --render   # HM1 whole-body+scale
python src/hm4_calib_sync.py --render                                              # HM4 camera-IMU sync
MUJOCO_GL=osmesa python src/hm5_retarget.py --mano_dir <dir>/mano --render        # HM5 MANO->Allegro retarget
# HM6 real data (MediaPipe detect in its own venv -> our MANO/SMPL fit in the main env):
python src/hm6_detect.py         &&  python src/hm6_realfit.py --mano_dir <dir>/mano --render  # HM6 hands
python src/hm6_detect_body.py    &&  python src/hm6_bodyfit.py --model <dir>/smplh/SMPLH_MALE.pkl --render  # HM6 body
# HM6 §11.3 대조: HMR2.0(4D-Humans) 학습 회귀 SOTA. neutral SMPL(라이선스) + 별도 venv 필요:
python src/hmr2_run.py     # HMR2.0 vs 우리 SMPL (같은 실사진, CPU)
```
