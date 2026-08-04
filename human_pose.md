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

## 9. 데이터셋 (실데이터는 HM6)

HM0·HM1은 **시뮬레이션**이다 — GT SMPL을 샘플링해 렌더하고, 그 관측으로 복원한다(GT를 알기에 오차를 mm로 잴 수 있음). **실제 데이터셋 검증은 HM6**에서 하며, 그때 해당 데이터셋의 실제 프레임(우리가 돌린 결과)을 올린다. 계획 데이터셋:

- **[HOT3D](https://facebookresearch.github.io/hot3d/)** (Meta, Aria/Quest egocentric, MANO 손 주석) — egocentric 손+물체.
- **[AssemblyHands](https://assemblyhands.github.io/)** (Assembly101 기반, ego+exo 동기, 3M 이미지) — egocentric 손.
- 전신은 **monocular HMR2.0/4D-Humans** 를 인터넷 영상에 돌려 정성 검증.

(각 데이터셋의 라이선스·다운로드가 필요하며, HM6 진입 시 실제 예시 이미지/영상을 이 페이지에 추가한다.)

## 재현
```bash
# SMPL+H 모델(license 등록 다운로드)을 <dir>/smplh/SMPLH_MALE.pkl에 두고:
python src/hm_smpl_fit.py   --model <dir>/smplh/SMPLH_MALE.pkl --sweep --render   # HM0 body
python src/hm1_wholebody.py --model <dir>/smplh/SMPLH_MALE.pkl --sweep --render   # HM1 whole-body+scale
```
