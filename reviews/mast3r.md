# MASt3R (ECCV 2024)

> "매칭은 원래 3D 문제다 — 2D 유사도가 아니라 3D 재구성 위에 매칭을 접지(grounding)하라" — DUSt3R에 matching head를 얹어 극단 시점차 강건성과 픽셀급 정밀도를 동시에 잡은 논문.

- Vincent Leroy, Yohann Cabon, Jerome Revaud (NAVER LABS Europe), ECCV 2024
- arXiv: https://arxiv.org/abs/2406.09756
- 코드: https://github.com/naver/mast3r

## 한 줄 요약
> 매칭을 2D 이미지 평면의 유사도 문제가 아니라 3D 재구성의 부산물로 재정의한 DUSt3R에, dense local feature를 회귀하는 두 번째 head와 InfoNCE 매칭 손실을 추가하고, O(kWH) fast reciprocal matching + coarse-to-fine으로 고해상도 픽셀급 대응을 뽑는 3D-grounded 매처 — Map-free localization에서 기존 최고(LoFTR+KBR) 대비 VCRE AUC **30% 절대 개선**.

## 문제
image matching은 mapping, localization, navigation 등 모든 3D 비전 파이프라인의 핵심인데, 지금까지 사실상 전부 **2D 문제로** 다뤄져 왔다. 그러나 대응점의 정의 자체가 "같은 3D 점을 관측한 픽셀들"이고, 2D 픽셀 대응과 3D 상대 포즈는 epipolar matrix로 직결된 동전의 양면이다. 저자들이 드는 증거: 가장 어려운 벤치마크인 Map-free(참조 이미지 1장, 시점차 최대 180°)에서 top 매처 LoFTR조차 VCRE precision 34%에 그치는 반면, 매칭용으로 설계되지도 않은 3D 재구성 모델 **DUSt3R의 부산물 대응점이 리더보드 1위**였다.

그런데 DUSt3R의 pointmap에서 reciprocal matching으로 뽑은 대응은 극단 시점차에 극도로 강건하지만 **정밀도가 낮다**. 원문 분석(§3.2): (i) 회귀(regression)는 본질적으로 노이즈에 취약하고, (ii) DUSt3R는 매칭을 명시적으로 학습한 적이 없다. 또 하나의 병목: dense feature 간 naive reciprocal matching은 **O(W²H²)**로, feature 추출(네트워크 forward)보다 오래 걸리는 역설적 상황이 된다.

## 방법
DUSt3R 골격 위에 (1) matching head + InfoNCE 손실, (2) fast reciprocal matching, (3) coarse-to-fine을 얹는다 (Fig.2, 파란색이 신규 기여).

### 1. DUSt3R 복습과 metric 예측으로의 손실 수정
Siamese ViT 인코더 → cross-attention을 주고받는 두 개의 intertwined decoder → head가 카메라 1 좌표계의 pointmap 2장을 회귀:
```
X^{1,1}, C^1 = Head³ᴰ₁([H^1, H'^1]),   X^{2,1}, C^2 = Head³ᴰ₂([H^2, H'^2])
```
DUSt3R의 회귀 손실은 scale 불변이었다:
```
ℓ_regr(v, i) = ‖ (1/z) X_i^{v,1} − (1/ẑ) X̂_i^{v,1} ‖,   z, ẑ = 유효 3D 점들의 원점 평균거리
```
**MASt3R의 수정**: map-free localization 같은 용도는 **metric 예측**이 필요하므로, GT가 metric인 데이터에서는 예측 쪽 정규화를 끈다 — 즉 z := ẑ로 두어 ℓ_regr = ‖X − X̂‖/ẑ. confidence 가중 손실은 DUSt3R와 동일:
```
L_conf = Σ_v Σ_i C_i^v · ℓ_regr(v, i) − α log C_i^v
```

### 2. Matching head + InfoNCE 매칭 손실
디코더 출력에 두 번째 head를 붙여 d=24 차원 dense local feature map을 회귀한다:
```
D^1 = Head_desc¹([H^1, H'^1]),   D^2 = Head_desc²([H^2, H'^2])
```
head는 GELU를 끼운 2-layer MLP, 출력은 unit norm. GT pointmap의 reciprocal 대응 M̂ = {(i,j) | X̂_i^{1,1} = X̂_j^{2,1}}에 대해 **양방향 InfoNCE** (cross-entropy) 손실:
```
L_match = − Σ_{(i,j)∈M̂} [ log s_τ(i,j) / Σ_{k∈P¹} s_τ(k,j)  +  log s_τ(i,j) / Σ_{k∈P²} s_τ(i,k) ]
s_τ(i,j) = exp[−τ D_i^{1⊤} D_j^2],   τ = 온도 (0.07)
```
핵심 포인트(원문): 이건 회귀가 아니라 **분류(classification) 손실**이라, "근처 픽셀"이 아니라 "정확히 그 픽셀"을 맞혀야만 보상된다 → 고정밀 매칭을 강하게 유도. 최종 목적함수:
```
L_total = L_conf + β L_match,   β = 1
```

### 3. Fast reciprocal matching (FRM)
목표는 상호 최근접(mutual NN) 집합 M = {(i,j) | j = NN₂(D_i¹), i = NN₁(D_j²)}. naive는 O(W²H²)이고 d=24 고차원에선 K-d tree도 무력하다. FRM은 **서브샘플 시드에서 출발하는 반복 NN 사상**:
- I¹의 규칙 격자에서 k개 픽셀 U⁰를 샘플 → NN으로 I²에 사상해 V¹ → 다시 I¹로 사상해 U¹ → 반복: `Uᵗ → [NN₂(D¹_u)] ≡ Vᵗ → [NN₁(D²_v)] ≡ Uᵗ⁺¹`
- 사이클을 이룬 쌍(U_n^t = U_n^{t+1})이 reciprocal match — 수렴한 점은 제거하고 나머지만 계속. 대부분 **5회 반복 안에 수렴**(Fig.3 중앙). 출력 M_k = ∪_t M_k^t.
- **복잡도 O(kWH)** — naive 대비 WH/k ≫ 1배 빠름. k=3000이면 **64배 가속**.
- **이론 보장**(appendix B): NN 방향 그래프의 각 부분그래프에는 사이클이 정확히 하나(Prop B.1, 유사도 단조증가 논증)이고 모든 시작점은 그 사이클로 수렴(Cor B.3), |M_k| ≤ k (Prop B.4).
- **정확도가 오히려 오르는 이유**: 시드가 큰 convergence basin으로 편향 샘플링돼 매칭의 공간 커버리지가 균질해지고, RANSAC의 epipolar 추정이 안정된다(appendix B.2, Fig.10). 실제로 random 서브샘플은 성능 붕괴, FRM은 full set보다 높음(Fig.12).

### 4. Coarse-to-fine 매칭
attention이 면적에 제곱이라 MASt3R는 최대 변 512px만 처리 → 고해상도(1M px)는 다운스케일 매칭 시 localization/재구성 품질이 크게 손상. 해법: (1) 다운스케일 영상에서 coarse 매칭 M_k⁰ → (2) 각 원본 영상에 512px 윈도 크롭 격자(50% 중첩) 생성 → (3) coarse 대응의 **90%를 덮을 때까지** 윈도 쌍을 greedy 추가 → (4) 윈도 쌍마다 독립적으로 MASt3R + FRM → (5) 원본 좌표로 환원해 병합.

### 5. 학습
- 14개 데이터셋 혼합(Habitat, ARKitScenes, BlendedMVS, MegaDepth, ScanNet++, CO3D-v2, Waymo, Map-free, WildRGB, VirtualKitti, Unreal4K, TartanAir 등) — 이 중 **10개가 metric GT**.
- 백본은 DUSt3R와 동일(ViT-Large 인코더 + ViT-Base 디코더), **DUSt3R 체크포인트로 초기화** 후 35 epoch fine-tune. epoch당 650k쌍, AdamW lr 1e-4 cosine, batch 64.
- 최대 변 512px에 종횡비 랜덤화(512×384/336/288/256/160) + **공격적 random crop**(principal point 중심을 보존하는 homography 변환) — coarse-to-fine이 줌인 크롭에서 동작하므로 학습 때 다양한 스케일 노출이 필수.
- 매칭 손실용 GT 대응은 pair당 4096개 샘플, 부족하면 false 대응으로 패딩.
- 추론 NN: 3D 점 매칭엔 K-d tree, d=24 feature엔 FAISS.

## 결과 (원문 수치)

### Map-free localization (핵심 벤치)
테스트셋(Table 2, VCRE<90px / Pose):
| Method | Reproj.↓ | Prec.↑ | VCRE AUC↑ | Med.Err.↓ | Pose AUC↑ |
|---|---|---|---|---|---|
| SIFT | 222.8px | 25.0% | 0.504 | 2.93m 61.4° | 0.252 |
| SP+SG | 160.3px | 36.1% | 0.602 | 1.88m 25.4° | 0.346 |
| LoFTR (KBR depth) | 165.0px | 34.3% | 0.634 | 2.23m 37.8° | 0.295 |
| DUSt3R | 116.0px | 50.3% | 0.697 | 0.97m 7.1° | 0.394 |
| **MASt3R (자체 metric depth)** | **48.7px** | **79.3%** | **0.933** | **0.36m 2.2°** | 0.740 |

VCRE AUC 93.3% — 2위 published LoFTR+KBR(63.4%) 대비 **30% 절대 개선**, 중앙 translation 오차 약 2m → **36cm**. 검증셋 ablation(Table 1): DUSt3R 3D점 매칭 0.704 → MASt3R feature 매칭 0.752 → 자체 metric depth 사용 시 **0.934**. 매칭 손실만 단독 학습(III)하면 중앙 회전오차 10.8°로 악화(회귀+매칭 병행 IV는 3.0°) — 디코더 용량을 매칭에 전부 줘도 이러하므로, **3D grounding 자체가 매칭 정밀도에 결정적**이라는 저자 결론.

### 상대 포즈 (Table 3, 10 views)
- **CO3Dv2**: RRA@15 94.6 / RTA@15 **91.9** / mAA(30) **81.8** — DUSt3R pairwise(94.3/88.4/77.2), DUSt3R-GA(96.2/86.8/76.7), PoseDiff(80.5/79.8/66.5) 대비 translation·mAA 우위.
- **RealEstate10k**: mAA(30) **76.4** — 최고 multi-view 방법 대비 +8.7pt, pairwise DUSt3R(61.2) 대비 +15.2pt. SfM 계열(COLMAP+SPSG 45.2)은 wide baseline에서 크게 밀림.

### Visual localization (Table 4)
- **Aachen Day-Night** (0.25m,2°/0.5m,5°/5m,10°): MASt3R top40 Day 82.2/93.9/99.5, Night 75.4/**91.6**/**100** — SOTA와 대등.
- **InLoc**: top40 DUC1 **56.1/79.3/90.9**, DUC2 **71.0/87.0/91.6** — LoFTR(47.5/72.2/84.8, 54.2/74.8/85.5), DKM 등 기존 SOTA를 큰 폭 상회. top1(검색 영상 1장)에서도 DUC1 41.9/64.1/73.2로 동작 — 3D grounding의 강건성. 단 매칭 없는 direct regression은 대형 씬에서 붕괴(Aachen Day top1 1.5/4.5/60.7) — 큰 공간일수록 feature matching이 필수.

### DTU MVS (Table 3 우측, zero-shot)
DTU 학습 없이 대응점 삼각측량만으로: acc 0.403 / comp 0.344 / **overall Chamfer 0.374mm** — DUSt3R(1.741) 대비 약 4.7배 개선, DTU로 학습한 GeoMVSNet(0.295)에 근접. zero-shot으로 이 수준은 최초라는 게 저자 주장. coarse-to-fine을 끄면 0.622로 저하(Table 5), Aachen Night top1도 최대 15%p 하락 — c2f의 기여 확인.

## 내 실습 연결
석사에서 대형 선체블록 wide-baseline 대응을 풀 때 SuperGlue/DKM/LoFTR 앙상블 → guided mask → COTR 정제 파이프라인으로 reprojection 29.1 → 2.23px까지 내렸다. 그 과정의 실패 분석이 정확히 이 논문의 문제의식이었다: **기하 없이 2D 유사도만 보는 매칭은 반복 패턴(용접선·격자 보강재)과 저텍스처 표면에서 무너진다**. 당시 나는 epipolar 기반 guided mask로 기하를 후처리 단계에서 주입해 이를 보완했는데, MASt3R는 같은 통찰을 반대 방향에서 실행한다 — 기하를 필터가 아니라 **표현 학습 단계에 내장**시켜, feature 자체가 3D 재구성과 같은 디코더에서 나오게 한다. ablation III→IV(매칭 단독 학습 시 포즈 3.0°→10.8° 악화)가 이 설계의 정당성을 수치로 보여준다.

- 내 파이프라인에서 COTR가 맡던 픽셀급 정제 역할을 MASt3R는 InfoNCE의 분류 손실("정확히 그 픽셀만 보상")이 학습 단계에서 해결한다. 후처리 단계 수를 줄일 수 있는 구조.
- 앙상블에서 체감한 상보성 — LoFTR가 커버리지, SuperGlue가 고신뢰 anchor — 를 MASt3R는 단일 모델의 pointmap(강건성) + descriptor(정밀도) 이중 head로 통합했다. Map-free의 LoFTR 0.634 → MASt3R 0.933 갭이 극단 시점차에서의 여유를 보여준다.
- FRM의 convergence basin 편향 샘플링이 커버리지를 균질화해 RANSAC을 안정화한다는 분석(appendix B.2)은, 내가 매칭 밀집 영역 때문에 epipolar 추정이 흔들리던 경험과 정확히 대응된다.
- 로보틱스 연결: camera-to-robot guidance에서 참조 뷰 1장으로 metric 상대 포즈가 나오는 것(Map-free 셋업 그대로)은 마커 없는 물체·카메라 정렬에 바로 쓸 수 있고, 자체 metric depth로 스케일 모호성이 없다는 점이 결정적이다.

## 한 줄 평 / 한계
"대응점의 정의는 3D인데 왜 2D에서 푸는가"라는 원리적 질문을 DUSt3R 위의 head 하나 + 손실 하나로 실행하고, 매칭·포즈·localization·MVS 네 과제를 동시에 밀어올린 논문 — 수렴 증명까지 붙인 FRM은 dense 매칭 실무의 표준 도구가 될 만하다.

한계:
- pairwise 전용 — 본 논문은 DUSt3R의 global alignment를 쓰지 않아, 다뷰 일관성은 후속(MASt3R-SfM 등)의 몫.
- 최대 변 512px 제약을 coarse-to-fine으로 우회하지만, 윈도 쌍마다 forward가 돌아 고해상도에서는 여전히 무겁다(실시간 SLAM 프론트엔드용은 아님).
- metric 예측은 metric GT를 가진 10개 데이터셋 분포에 의존 — 분포 밖 스케일(초근접 소형 물체 등)의 신뢰도는 검증 필요.
- 작은 씬에서는 direct regression이 매칭과 대등하지만 큰 씬에서 붕괴 — 씬 규모에 따라 매칭/회귀 경로를 골라야 하는 이원성이 남는다.
