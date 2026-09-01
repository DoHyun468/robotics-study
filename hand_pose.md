# 손 자세 추정 — 멀티뷰 MANO 피팅 (실측)

파라메트릭 **손 모델(MANO)** 을 우리 캘리브레이션 스택 위에서 직접 다뤄 본 실측이다.

**한눈에:** E1–E5는 **손 자체를 카메라로 3D 이해**하는 지각 실험(자세·모양·오차 — 물체를 집는 게 아니다), E6는 그렇게 읽은 **사람 손 파지를 로봇 팔이 따라 집게** 하는 실행 실험이다.

방법론은 이 사이트의 [perception 파이프라인](perception.md)과 동일 — **GT를 알고, 관측에서 복원하고, 오차를 mm로 잰다** — 단 대상이 강체 6-DoF pose에서 **관절체 파라메트릭 모델**(shape β + pose)로 한 칸 커졌다. 모델 자체 리뷰는 [SMPL·MANO](reviews/smpl-mano.md).

## 방법

1. GT 손 샘플: shape $\beta\in\mathbb{R}^{10}$ + pose(global orient + PCA 계수)를 뽑아 MANO forward → 3D 관절·메시.
2. **N개 캘리브 카메라**(고정 intrinsic $K$ + 링 배치 extrinsic)에 3D 관절을 투영 → 2D 키포인트 + **픽셀 노이즈** 주입(=검출 노이즈 모사).
3. 복원: $(\beta,\text{pose})$ 를 0에서 시작해 **멀티뷰 reprojection 오차**를 Adam으로 최소화(약한 shape prior 포함).
4. GT 대비 복원 손의 **관절 오차(mm)** · 버텍스 오차 · reprojection(px) · $\beta$ 오차를 5-seed 평균으로 측정.

:::{dropdown} 해부 — 피팅 한 사이클: 입력→최적화→출력 (`src/mano_fit.py` 발췌)
**분류가 중요하다**: 이건 학습(learned regression)이 아니라 **인스턴스별 gradient 최적화**다 — 사전학습 가중치가 없고, MANO forward가 미분가능하다는 점만 이용해 매 손마다 처음부터 푼다. **입력**: 뷰별 캘리브 `(K 3×3, R 3×3, t 3)` × N + 2D 관측 `(21,2)` px × N. **출력**: `β (10,)` + global orient `(3,)` + pose PCA `(n,)` → forward하면 3D 관절 `(21,3)` [m].

```python
def project(P3, K, R, t):                       # 관절체 버전의 핀홀 투영 (미분가능)
    Pc = P3 @ Rt.T + tt                         # world → camera
    uvw = Pc @ Kt.T
    return uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-6)   # 동차 나눗셈 → (N,2) px

# 최적화 대상 3개를 0에서 시작 (GT를 훔쳐볼 수 없음)
betas = torch.zeros(1, 10, requires_grad=True)
go    = torch.zeros(1, 3, requires_grad=True)          # global orient
pose  = torch.zeros(1, n_pca, requires_grad=True)      # PCA 계수 = 자유도 손잡이
opt = torch.optim.Adam([betas, go, pose], lr=0.05)
for it in range(iters):
    J, _ = joints_of(model, betas, go, pose)           # MANO forward → 3D 관절
    loss = 0.0
    for (K, R, t), uv_obs in zip(cams, obs):           # 모든 뷰의 재투영오차 합
        uv = project(shift(J), K, R, t)
        loss = loss + ((uv - uv_obs) ** 2).sum(-1).mean()
    loss = loss / len(cams) + 1e-3 * (betas ** 2).mean()   # 약한 shape prior
    loss.backward(); opt.step()
```

E4–E8은 전부 이 루프의 **관측 또는 목적함수만 바꾼 변주**다: E4 = 관측에 표면 대응점 200개 추가, E5 = 피팅에 쓰는 `(K,R,t)`만 섭동(관측은 참 카메라), E7 = `uv_obs` 일부를 무작위 마스킹, E8 = 프레임 간 `‖poseₜ−poseₜ₋₁‖²` 항 추가. 실험 축이 코드에서 한 줄씩이라는 것 자체가 이 방법론(GT 알고 복원 재기)의 힘이다.
:::

<video src="_static/mano_fit.mp4" controls loop muted playsinline style="width:100%;max-width:520px;border-radius:8px"></video>

*cam 1 시점에서 최적화가 수렴하는 과정 — 초기(0에서 시작)에서 GT(teal 원)로 복원값(주황 ×)이 붙는다.*

<img src="_static/mano_multiview.png" alt="multi-view MANO fitting" style="width:100%;max-width:960px;border-radius:8px">

*4개 뷰의 최종 피팅 — GT 관절(teal)과 복원 관절(주황 ×)이 거의 겹친다(관절 복원오차 0.47mm). 회색 점은 재투영된 손 메시(778버텍스).*

## 결과 (5-seed 평균)

<img src="_static/mano_sweep.png" alt="recovery-error sweeps" style="width:100%;max-width:960px;border-radius:8px">

**① 뷰 수** (노이즈 1px, PCA 15)

| 뷰 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 관절 복원오차 | 1.27mm | 0.99mm | **0.46mm** | 0.39mm |

단일 뷰는 depth 모호성으로 최악(1.27mm), 4뷰에서 0.46mm로 −64%, 그 뒤로는 포화. 우리 [depth A/B](perception.md)에서 "단안의 약제약을 멀티뷰 기하가 푼다"와 같은 곡선이 관절체에서도 나온다.

**② 픽셀 노이즈** (4뷰, PCA 15)

| 노이즈 | 0px | 1px | 2px | 4px |
|---|---|---|---|---|
| 관절 복원오차 | 0.27mm | 0.46mm | 0.77mm | 1.45mm |

2D 검출 노이즈가 3D 오차로 **거의 선형 전파**(reprojection도 0.38→4.54px로 동반). [perception 오차예산](perception.md)의 "입력 오차 → 출력 오차 전파"가 파라메트릭 피팅에서도 성립.

**③ pose-PCA rank = 정규화** (4뷰, 1px)

| PCA rank | 6 | 10 | 15 | 45 |
|---|---|---|---|---|
| 관절 복원오차 | 0.38mm | 0.45mm | 0.46mm | 0.51mm |
| **버텍스 오차** | **0.58mm** | 0.93mm | 1.49mm | **11.06mm** |

핵심 발견: pose PCA 차원을 늘릴수록(자유도↑) **노이즈에 과적합**해 버텍스 오차가 11mm로 폭발한다. 적은 차원(6)이 오히려 최선 — [MANO의 pose PCA](reviews/smpl-mano.md)(6/10/15개=81/90/95%)가 그 자체로 **정규화기**로 작동함을 실측으로 확인.

**④ shape β는 키포인트로 안 잡힌다**: 모든 조건에서 $\beta$ 오차가 1.5–2.7로 크게 남는다. **키포인트는 pose를 제약하지 shape를 제약하지 않는다** — 체형 복원엔 실루엣·표면 관측이 필요하다는 정직한 한계(→ 다음 실험 E4).

## 연결

- **강체 → 관절체**: [markerless ICP](perception.md)가 "형상 모델을 알면 부분 관측으로 6-DoF pose가 잡힌다"였다면, 이건 "**변형 가능한** 형상 모델을 알면 관절 자세까지 잡힌다"로의 확장. 멀티뷰·노이즈·정규화의 구조가 동일하게 재현된다.
- **파라메트릭 모델 경험**을 읽은 것([SMPL·MANO](reviews/smpl-mano.md))에서 **직접 측정한 것**으로 옮긴 실측이다. 최적화 피팅의 반대편(회귀 기반 단안 복원)은 [HaMeR](reviews/hamer.md).

## 재현

```bash
# MANO 모델 파일(license 등록 다운로드)을 <MANO_dir>에 두고:
python src/mano_fit.py --mano_dir <MANO_dir> --sweep --render
```

## E4 — shape(β)는 표면 관측이 있어야 잡힌다

E1에서 "키포인트로는 shape가 안 잡힌다"를 봤다. 이걸 대조 실험으로 확인한다: 관측을 (a) **관절 키포인트만** vs (b) **관절 + 표면 대응점 200개**(실루엣/표면이 주는 정보의 대리)로 두고 각각 피팅.

<img src="_static/mano_e4.png" alt="shape observability" style="width:100%;max-width:520px;border-radius:8px">

| 관측 | 관절오차 | **버텍스오차** | **β 오차** |
|---|---|---|---|
| joints only | 0.46mm | 1.42mm | 1.71 |
| **+ surface (200pt)** | 0.37mm | **0.37mm (−74%)** | **1.11** |

표면 대응점을 더하면 버텍스 오차가 1.42→0.37mm로 무너지고 β 오차도 준다. **키포인트는 pose를, 표면은 shape를 제약한다** — 체형까지 복원하려면 실루엣·표면 항이 필요하다는 걸 실측으로 확인. (β는 본질적으로 어려워 완전히 0은 안 되지만 방향은 뚜렷.)

## E5 — 캘리브 오차가 손 관절오차로 전파

관측은 **참(true) 카메라**로 만들고, 피팅만 **섭동된(miscalibrated) 카메라**로 하면, 캘리브 오차가 3D 복원오차로 얼마나 새는지 측정된다 — [perception 오차예산](perception.md)의 관절체판.

<img src="_static/mano_e5.png" alt="calibration error propagation" style="width:100%;max-width:960px;border-radius:8px">

| 섭동 | baseline | 회전 0.5°/1°/2° | 위치 2/5/10mm | 초점 0.5/1/2% |
|---|---|---|---|---|
| 관절 복원오차 | 0.44mm | 0.56 / 0.82 / 1.43mm | 2.18 / 5.47 / **10.96mm** | 0.54 / 0.79 / 1.4mm |

**카메라 위치 오차는 거의 1:1로 전파**(10mm 캘리브 오차 → 10.96mm 관절오차), 회전·초점도 단조 증가. [perception 파이프라인](perception.md)에서 "정확도 병목은 제어가 아니라 캘리브·기하"라 한 결론이 파라메트릭 손 피팅에서도 그대로 — **캘리브가 데이터·복원 품질의 상한**이다.

## E7 — 자기가림(self-occlusion) 강건성

손은 손가락끼리 서로 가려 관절이 안 보이는 경우가 많다(자기가림). 뷰마다 키포인트 일부를 무작위로 **가리고**(관측에서 제거) 피팅한다.

<img src="_static/mano_e78.png" alt="E7 occlusion and E8 temporal results" style="width:100%;max-width:1000px;border-radius:8px">

*좌·중 = E7(자기가림) · 우 = E8(시간축 추적).*

| 가림 비율 | 0% | 20% | 40% | 60% |
|---|---|---|---|---|
| 관절 복원오차 | 0.41mm | 0.49mm | 0.65mm | 0.94mm |

완만하게 열화하고, **멀티뷰가 가림을 강하게 보완**한다 — 가림 40%에서 1뷰 **2.69mm** → 4뷰 0.65 → 8뷰 **0.45mm**. 한 뷰에서 가려진 관절이 다른 뷰에서 보이면 복원된다. 손 자기가림에 멀티뷰가 정공법이라는 실측.

## E8 — 시간축 손 추적 (temporal)

움직이는 손(24프레임)을 **(a) 프레임별 독립 피팅** vs **(b) 시간 스무딩**(프레임 간 pose 변화를 penalize하는 공동 최적화)으로 복원.

| | 관절 오차 | 지터(프레임 간 떨림) |
|---|---|---|
| 프레임별 | 0.61mm | 0.46mm |
| **시간 스무딩** | **0.57mm** | **0.41mm** |

시간 스무딩이 지터를 줄인다(노이즈가 클수록 효과↑) — 손 추적에서 시간 일관성 항의 값. 실시간 텔레옵([H2O](reviews/h2o.md))이 30Hz pose 추정을 쓰는 이유와 같은 맥락.

## E6 — MANO → 로봇 그리퍼 retargeting → 물리 pick

사람 손 파지를 로봇 action으로 옮기는 단계([H2O](reviews/h2o.md)/[UMI](reviews/umi.md) 축). **손(MANO)은 아무것도 집지 않는다** — 손 자세에서 "어떻게 잡을지"(개폐폭·각도)만 읽고, **실제로 박스를 집는 건 로봇 팔(Franka Panda)** 이다.

<img src="_static/e6_pipeline.png" alt="E6 pipeline MANO grasp to gripper to Panda pick" style="width:100%;max-width:1000px;border-radius:8px">

*① MANO 손 파지에서 개폐폭·각도를 읽음 → ② 그리퍼 명령으로 변환 → ③ 로봇 팔이 그 파지로 박스를 집음.*

MANO 파지에서 **개폐폭**(thumb–index span)과 **grasp yaw**(opening 축)를 뽑아 평행 그리퍼 명령으로 매핑하고, 실제로 박스를 집어 든다.

:::{dropdown} 해부 — 손 자세에서 그리퍼 명령 3개를 뽑는 매핑 (`src/mano_retarget.py` · `mano_retarget_panda.py` 발췌)
**입력**: MANO 3D 관절 `(21,3)`. **출력**: 스칼라 3개 — aperture [m] · yaw [rad] · 그리퍼 프레임 `R`. 전부 룰베이스 기하:

```python
thumb, index, wrist = J[THU_TIP], J[IDX_TIP], J[0]
aperture = np.linalg.norm(thumb - index)      # ① 개폐폭 = 엄지팁–검지팁 거리
center = 0.5 * (thumb + index)
y = (index - thumb) / ‖·‖                     # ② opening 축 (손가락이 벌어진 방향)
a = (center - wrist) / ‖·‖                    # ③ approach 축 (손목 → pinch)
z = a - (a @ y) * y                           #    직교화 → 그리퍼 z
R = np.stack([np.cross(y, z), y, z], axis=1)  #    cols = [x, y(open), z(approach)]

# panda 실행 쪽 (mano_retarget_panda.py): 스칼라 2개가 파이프라인에 꽂히는 지점
yaw = atan2(y[1], y[0])                       # opening 축을 테이블 평면에 투영
box_w = min(0.068, max(0.030, ap * 0.55))     # 손이 쥐었을 물체 크기 추정 (72mm 게이트)
Rg = bp.topdown_R(yaw)  →  dls_ik  →  sim     # 이후는 manipulation 공통 파이프라인 그대로
```

"매핑은 쉽다"의 실체가 이 몇 줄이다 — 21관절 손 자세가 결국 스칼라 2개(yaw·aperture)로 압축되고, 어려움은 전부 그 뒤의 접촉 물리 실행(0/12 vs 10/10)에 있었다.
:::

**매핑은 쉽다 — 물리 실행이 어렵다.** 두 가지 실행을 비교했다:

| 실행 방식 | 결과 | 비고 |
|---|---|---|
| 자체 플로팅 그리퍼(직접 작성) | **0/12** | 매핑(개폐폭 MAE 3.71mm)은 맞으나, lift 시 **파지가 미끄러져 박스 이탈**(접촉·grip 튜닝 실패) |
| **검증된 Franka Panda pick 스택 재사용** | **10/10** | MANO yaw+aperture를 우리 [매니퓰레이션](manipulation.md) pick(DLS IK + PD + 접촉물리)에 주입 → 매번 파지·인상 성공(172–200mm) |

<video src="_static/mano_retarget_panda.mp4" controls loop muted playsinline style="width:100%;max-width:720px;border-radius:8px"></video>

*MANO 파지에서 유도한 yaw·개폐폭으로 Franka Panda가 박스를 집어 든다 (10/10).*

**교훈(정직):** retargeting의 기하 매핑(손 파지→그리퍼 개폐·자세)은 어렵지 않다. 진짜 난관은 **접촉 물리로 실제 들어올리는 실행**이고, 이건 co-tune된·검증된 제어 스택이 있어야 풀린다 — 우리가 이미 실측으로 쌓은 [매니퓰레이션](manipulation.md) pick이 정확히 그 역할을 했다. (자체 그리퍼의 실패도 그대로 기록: 안 되는 건 안 된다고.)

## 정리

강체 6-DoF([perception](perception.md)) → **관절체 파라메트릭 손**(E1–E5) → **사람 손→로봇 action**(E6)으로 이어지는 실측 사슬. 파라메트릭 모델 피팅(멀티뷰·정규화·오차전파·shape 관측성)과 human-to-robot retargeting을 읽은 것([SMPL·MANO](reviews/smpl-mano.md)·[H2O](reviews/h2o.md))에서 **직접 측정한 것**으로 옮겼다.
