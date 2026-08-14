# Human Motion → Robot Data

사람의 몸·손 동작을 **캡처하고(파라메트릭 인체 모델), 로봇의 몸으로 옮기고(retargeting), 로봇 학습 데이터로 바꾸는(demonstration 파이프라인)** 계열 리뷰. 로봇 시연 데이터가 비싸다는 병목을 "사람의 동작"이라는 무한한 원천으로 푸는 축이다 — [VLA](vla.md)가 소비하는 데이터의 **상류 공급망**이라고 보면 정확하다.

## 파이프라인 한 장

```
사람 동작 (멀티카메라/에고센트릭/핸드헬드)
   │  3D pose 추정 + 파라메트릭 모델 피팅 (SMPL·MANO)     ← 캡처
   ▼
사람 기준 동작 표현 (관절각·궤적)
   │  kinematics 대응 + 최적화 (shape 피팅·feasibility 필터) ← retargeting
   ▼
로봇 기준 동작 (joint targets / EE 궤적)
   │  시연 데이터셋화 + 정책 학습 (BC/diffusion/VLA)        ← 데이터 파이프라인
   ▼
로봇 실행
```

**캡처 — 파라메트릭 모델 & 3D pose**
- [SMPL · MANO (2015·2017)](smpl-mano.md) — 몸(6890버텍스)·손(778버텍스)의 **파라메트릭 모델**. 이 계열 전체의 공통 언어.
- [SMPL-X (2019)](smplx.md) — 몸+손+얼굴을 하나로 합친 통합 파라메트릭 모델(SMPL+MANO+FLAME) + SMPLify-X 최적화 피팅.
- [HaMeR (2024)](hamer.md) — 단안 RGB → **MANO 파라미터 회귀**(ViT). 영상에서 손을 바로 뽑는 캡처 프론트엔드.

**retargeting & 시연 수집**
- [H2O (2024)](h2o.md) — SMPL 피팅 → 휴머노이드 **retargeting** → RL 추적 정책 → **RGB 캠 하나로 실시간 전신 텔레옵**.
- [UMI (2024)](umi.md) — 핸드헬드 그리퍼로 **로봇 없이 시연 수집** → diffusion policy. 시연 데이터 파이프라인의 실용 최전선.
- [DexCap (2024)](dexcap.md) — UMI가 못 가는 **다지손** 영역: SLAM+EMF 장갑 모캡 리그($4k)로 가림에 강하게 캡처 → fingertip IK로 LEAP 핸드 retarget → point-cloud diffusion policy.

**상호작용 & 에고센트릭**
- [GRAB (2020)](grab.md) — 전신 인간의 **물체 파지·조작** MoCap(SMPL-X + 접촉). hand-object interaction의 데이터 기준.
- [HOT3D (2024)](hot3d.md) — **에고센트릭** 손·물체 3D 추적 데이터셋(Aria/Quest3). 1인칭 wearable 수집.

> 📐 위 파라메트릭 모델을 우리 캘리브 스택에서 **직접 피팅해 본 실측**은 [손 자세 추정 — 멀티뷰 MANO 피팅](../hand_pose.md).

## 비교표

| 리뷰 | 연도 | 파이프라인에서의 위치 | 핵심 수치 (원문 대조) |
|------|------|----------------------|----------------------|
| [SMPL·MANO](smpl-mano.md) | 2015·2017 | **캡처의 공통 언어** — 저차원 파라미터로 몸·손을 표현 | SMPL 6890버텍스·24관절, MANO 778버텍스·PCA 6/10/15개=81/90/95% |
| [H2O](h2o.md) | 2024 | **retargeting + 실시간 텔레옵** | AMASS 13k 시퀀스 → retarget 10k → feasibility 필터 ~8.5k, HybrIK 30Hz·H1 19-DoF |
| [UMI](umi.md) | 2024 | **시연 수집 장치 + 정책** (retargeting을 장치 설계로 회피) | 그리퍼 BoM $73, SLAM 오차 6.1mm/3.5°, in-the-wild 71.7% |
| [DexCap](dexcap.md) | 2024 | **다지손 시연 수집** (추적을 센서로 강화 + fingertip IK) | 리그 $4k·1.8kg, 텔레옵 3배 속도, 기본 3태스크 평균 72% |

## 이 갈래를 읽는 축

- **사람 동작을 쓰는 세 전략**: H2O는 사람 몸 → 로봇 몸으로 **명시적 retargeting**(kinematics 대응·최적화)을 하고, UMI는 애초에 **로봇과 같은 end-effector를 사람 손에 들려서** retargeting 문제 자체를 장치로 소거한다. DexCap은 그 중간 — 소거가 불가능한 **다지손**에서 추적을 센서 물리(EMF+SLAM)로 강화하고 retargeting을 fingertip IK 한 층으로 얇게 만든다. 전신은 H2O, 그리퍼 조작은 UMI, 다지손은 DexCap이 각각의 답.
- **캘리브레이션·기하가 전 구간의 바닥**이다: 멀티카메라 pose 추정은 카메라 간 시공간 캘리브레이션 위에서만 성립하고, UMI의 SLAM 오차(6.1mm)는 곧 시연 데이터의 라벨 노이즈가 된다 — 이 사이트 [perception 파이프라인](../perception.md)에서 실측한 "캘리브 오차의 하류 전파"와 정확히 같은 구조.
- [VLA](vla.md)와의 관계: VLA는 (관측, 행동) 시연을 소비한다. 이 갈래는 그 시연을 **사람에게서 값싸게 만드는** 쪽이다 — OpenVLA가 쓴 Open X-Embodiment(로봇 텔레옵 수집)의 병목을 우회하는 다음 세대 공급망.
