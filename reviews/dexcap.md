# DexCap (2024)

*로봇 없이 **다지손** 시연을 모은다 — SLAM + 전자기장 장갑 모캡 리그와 DexIL 파이프라인*

Wang, Shi, Wang, Zhang, Fei-Fei, Liu, *DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation*, RSS 2024 (arXiv 2403.07788).

카테고리 개요: [Human Motion → Robot Data](humanmotion.md) · 비교축: [UMI](umi.md)(장치 설계로 retargeting 소거) · [H2O](h2o.md)(전신 명시적 retargeting)

## 한 줄 요약

[UMI](umi.md)는 사람에게 로봇 그리퍼를 들려 retargeting을 소거했다 — 그러나 그 트릭은 end-effector가 평행조 그리퍼일 때만 성립한다. DexCap은 **그 트릭이 불가능한 다지손 영역**의 답: 사람 맨손을 그대로 두고, 가슴 LiDAR 카메라 + 손등 SLAM 카메라 + **전자기장(EMF) 장갑**으로 가림(occlusion)에 강한 손목 6-DoF·손가락 추적을 확보한 뒤, **fingertip IK 한 층**으로 LEAP 핸드에 옮겨 point-cloud diffusion policy(DexIL)를 학습한다. 리그 $4k, 텔레옵 대비 3배 수집 속도, 다지 태스크 6종 — 그리고 정책 실행 중 사람 손 델타를 잔차로 섞는 human-in-the-loop 보정까지.

---

## 1. 문제 — 다지손 시연은 양쪽 다 막혀 있었다

다지 핸드(16+ DoF) 정책 학습의 데이터 병목은 그리퍼보다 훨씬 심하다:

1. **텔레옵이 특히 어렵다** — 사람 손 → 로봇 손의 대응을 실시간으로 풀며 조작해야 해서 느리고, 미세한 손가락 협응은 화면 너머로 재현이 안 된다.
2. **비전 기반 손 추적([HaMeR](hamer.md)류)은 가림에 약하다** — 조작 데이터에서 가장 중요한 순간(손이 물체를 감싸는 순간)이 정확히 손가락이 가려지는 순간이다. 라벨이 가장 필요한 프레임에서 라벨러가 죽는다.
3. **UMI 트릭은 못 쓴다** — "로봇의 손을 사람에게 들려준다"는 EE가 단순할 때의 해법. 다지손을 들려줄 수는 없다.

> **DexCap의 답**: retargeting을 소거할 수 없다면, **추적을 센서 물리로 강화**한다. 손가락은 빛이 아니라 **전자기장**으로 읽어(가림 무관) 캡처 문제를 풀고, 남은 사람↔로봇 격차는 fingertip IK 한 층으로만 흡수한다.

## 2. 방법

### 2.1 리그 — $4k, 1.8kg, 가림 내성

| 부품 | 역할 |
|------|------|
| Intel RealSense **L515** (가슴) | 관측 RGB-D(LiDAR) — 정책 입력 포인트클라우드의 원천 |
| Intel RealSense **T265 ×2** (양 손등) | 손목 6-DoF를 **각자 SLAM으로** 추적 |
| **기준 T265** (L515 아래) | 초기 pose가 **월드 프레임**을 정의 |
| **Rokoko EMF 장갑** | 손가락 관절 — 전자기장 방식이라 시각 가림과 무관 |
| NUC 13 Pro + 40,000mAh 배터리 (백팩) | 기록 — 1회 충전 ~40분 수집 |

총 3.96lb(≈1.8kg), 전체 예산 **$4k 이내**. 시각 손 추적과의 정성 비교에서 물체를 감싸쥔 상태에서도 손가락 자세가 유지됨을 보인다(가림 내성) — 다만 UMI의 6.1mm 같은 **정량 정밀도 수치는 없다**.

### 2.2 좌표 통일 — 모든 스트림을 월드로

기준 T265의 초기 pose를 월드 $W$로 잡고, 각 센서가 자체 추정한 pose로 모든 데이터를 월드에 정렬한다. 시각 $t$의 가슴 카메라 pose를 $T^W_C(t) \in SE(3)$, 손목 pose를 $T^W_{\text{wrist}}(t)$라 하면:

$$
P^W_t = T^W_C(t)\,P^C_t, \qquad
x^W_{\text{tip},i}(t) = T^W_{\text{wrist}}(t)\;x^{\text{wrist}}_{\text{tip},i}(t)
$$

- $P^C_t$: L515 깊이에서 나온 카메라 프레임 포인트클라우드, $x^{\text{wrist}}_{\text{tip},i}$: EMF 장갑이 주는 손목 기준 $i$번 손가락 끝 위치.
- 이 변환 한 줄이 결과의 절반을 설명한다: **관측자가 걸어다녀도(가슴 카메라가 움직여도) 데이터는 정지 좌표계에 놓인다.** 뒤의 ablation에서 in-the-wild 데이터로 학습한 이미지 정책이 0%로 무너질 때 포인트클라우드 정책이 사는 이유가 정확히 이 정렬이다.

### 2.3 DexIL — 사람 데이터 → 로봇 정책

**(1) Retargeting = fingertip IK.** LEAP 핸드는 4지 16-DoF다. 사람 새끼손가락을 버리고, 나머지 네 손가락 끝점을 맞추는 IK로 관절각을 푼다:

$$
q^\star_t \;=\; \arg\min_{q\,\in\,\mathbb{R}^{16}} \sum_{i\,\in\,\{\text{엄지,검지,중지,약지}\}} \bigl\lVert \mathrm{FK}_i(q) - x^W_{\text{tip},i} \bigr\rVert_2^2
$$

$\mathrm{FK}_i$는 로봇 손 $i$번 손가락 끝의 forward kinematics. 실시간으로 부드러운 해를 내도록 구현했고, 손목 6-DoF는 IK의 초기 참조로 쓴다. [H2O](h2o.md)의 전신 retargeting(shape 피팅 + feasibility 필터)에 비하면 얇은 층이다 — 손끝 위치만 맞추면 되는 문제로 축소했기 때문.

**(2) 관측 표현.** RGB-D → 포인트클라우드, **5,000점 균일 다운샘플 + RGB 색 concat**. 테이블 등 태스크 무관 요소는 제거하고, 결정적 한 수로 — **로봇 손을 FK로 렌더한 점들을 관측에 합성**한다. 학습 데이터(사람 손)와 실행(로봇 손)의 embodiment 격차를 관측 쪽에서 미리 좁히는 장치다.

**(3) 행동 표현.** $a_t = s_{t+1}$ — 다음 상태(손목 pose + 16 관절각)가 곧 행동인 position control.

**(4) 정책.** Diffusion Policy + **Perceiver 인코더**(DP-perc). action chunk $A_t = (a_t,\dots,a_{t+d})$에 표준 DDPM 노이즈를 씌우고 관측 조건부로 복원한다:

$$
\mathcal{L} \;=\; \mathbb{E}_{k,\varepsilon}\,\bigl\lVert \varepsilon - \varepsilon_\theta\bigl(\sqrt{\bar\alpha_k}\,A_t + \sqrt{1-\bar\alpha_k}\,\varepsilon,\;k,\;o_t\bigr) \bigr\rVert^2
$$

($\bar\alpha_k$: 노이즈 스케줄 누적곱, $o_t$: 포인트클라우드 관측 임베딩.) 증강은 작업 공간 안 2D 랜덤 평행이동.

### 2.4 Human-in-the-loop 잔차 보정

배포한 정책이 실패하는 지점에서, 사람이 리그를 낀 채 **델타만** 얹는다. 사람 손의 초기 상태 대비 변화 $(\Delta p^H_t, \Delta J^H_t)$를 정책 출력 $(p_{t+1}, J_{t+1})$에 잔차로 합성:

$$
a'_t \;=\; \bigl(\,p_{t+1} \oplus \alpha\,\Delta p^H_t,\;\; J_{t+1} + \beta\,\Delta J^H_t\,\bigr)
$$

$\oplus$는 SE(3) 합성, $\alpha,\beta$는 보정 게인 — 원문 기준 $\beta < 0.1$의 작은 스케일이 최적 UX다(정책이 주도하고 사람은 미세 조향). 보정 시연 $D'$와 원본 $D$를 50:50으로 섞어 재학습한다. 풋 페달로 완전 텔레옵 모드 전환도 가능.

## 3. 결과 (원문 대조)

**6개 다지 태스크**: Sponge Picking / Ball Collecting / Plate Wiping(양손) / Packaging(양손, 학습 6물체 + 미학습 9물체) / Scissor Cutting / Tea Preparing(병뚜껑 열고 핀셋으로 찻잎 옮기기).

**아키텍처 ablation (3개 기본 태스크, 성공률):**

| 모델 | Sponge | Ball | Plate | 평균 |
|------|--------|------|-------|------|
| BC-RNN-img-mask | 25% | 10% | 10% | 15% |
| BC-RNN-point | 45% | 30% | 25% | 33% |
| DP-img-mask | 55% | 40% | 30% | 42% |
| DP-point | 75% | 65% | 50% | 63% |
| **DP-perc (최종)** | **85%** | **60%** | **70%** | **72%** |

읽는 법 세 줄: **① 이미지 정책은 손 마스킹 없이는 0%** — 사람 손이 찍힌 관측을 그대로 쓰면 실행 시 로봇 손과의 도메인 격차로 전멸한다. **② diffusion > BC-RNN ~25%p** — 다지 행동의 다봉성. **③ 포인트클라우드 > 이미지, Perceiver > PointNet**(양손 협응에서 ~20%p) — §2.2의 월드 정렬이 표현 선택과 맞물리는 지점.

**in-the-wild(Packaging, 걸어다니며 수집):** 이미지 계열 full-task **0%** vs 포인트클라우드 **47%**(학습 물체) / **40%**(미학습 9물체).

**HITL 보정(30회):** Scissor Cutting full-task 0% → **20%**(subtask 45%), Tea Preparing 뚜껑 열기 30% → **65%** — 실패 지점만 사람이 채워넣는 데이터 효율.

**수집 속도:** 텔레옵 대비 **3배** (Ball Collecting 기준), 자연스러운 맨손 동작에 근접.

## 4. 내 실습 연결

- **[Track B 파이프라인](../rldx.md)(§6 human 시연 → LeRobot v2.1)과 같은 문제의 실물 버전** — 우리는 "사람 데몬 → 로봇 학습 스키마" 변환을 합성 데몬으로 구축해 RLDX 로더 검증까지 갔는데, DexCap은 그 상류(실제 사람 손 캡처)를 하드웨어로 푼 것. 좌표 통일(§2.2)이 데이터 품질을 지배한다는 구조도 동일하다.
- **fingertip IK는 우리 [HM5 grasp 합성](../human_pose.md)과 같은 언어** — 우리도 Allegro(4지 16-DoF)로 사람 손 대응을 풀며 새끼손가락을 버렸다. DexCap의 "IK 한 층으로 morphology gap 흡수"가 어디까지 버티는지(힘·접촉은 전달 안 됨)는 우리 force-closure/침투 실험이 보는 바로 그 한계다.
- **[RLDX-1](rldx1.md)의 데이터 엔진과 대비** — RLDX mid-training은 자체 텔레옵(ALLEX)+DROID로 다지 데이터를 만든다. DexCap은 그 텔레옵 병목의 대안 공급망이다: 다지손 파운데이션 모델일수록 이 계열([UMI](umi.md)→DexCap)이 상류가 된다. [DexBench](../rldx.md)의 Contact Precision 축 태스크가 요구하는 데이터가 정확히 이것.
- **[HaMeR](hamer.md)·[HOT3D](hot3d.md)와의 관계** — 비전 단독 손 추적의 가림 문제를 DexCap은 센서로 우회했다. 반대로 비전 쪽이 가림을 이겨내면(에고센트릭 멀티뷰, 모델 프라이어) 장갑 없는 DexCap이 가능해진다 — 두 갈래가 수렴하는 지점이 "맨손 in-the-wild 다지 데이터"다.

## 5. 한 줄 평·한계

**한 줄 평.** "retargeting을 소거할 수 없는 곳에서는 **추적을 물리로 이긴다**" — 가림이라는 캡처 최대의 적을 EMF+SLAM으로 우회하고, 사람↔로봇 격차를 IK 한 층 + 관측 합성(FK 점 주입)으로만 흡수한 설계. UMI와 함께 "시연 수집을 로봇에서 분리한다"는 같은 강령의 다지손 판이다.

**한계.**
- **절대 성공률이 낮다** — 최고 85%, full-task는 47~70%, 가위질은 보정 없이는 0%. 다지 모방의 난이도가 수집을 풀어도 그대로 남는다는 정직한 수치.
- **힘·촉각 부재** — 궤적과 관절각만 기록한다. 접촉력이 성패를 가르는 태스크는 [3D-ViTac](3d-vitac.md) 계열의 센서 층이 따로 필요하다.
- **정밀도 정량 없음** — UMI의 6.1mm/3.5° 같은 라벨 노이즈 수치가 없어 데이터 품질의 바닥을 논문만으로 가늠하기 어렵다.
- **장비 부담** — $4k + 장갑·백팩 착용은 UMI($371)의 "아무나 어디서나"보다 무겁다. 4지 LEAP 전제(새끼손가락 폐기)도 손 형태 일반화의 제약.
