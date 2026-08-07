# DUSt3R: Geometric 3D Vision Made Easy (CVPR 2024)

*"매칭→essential matrix→삼각측량→BA"라는 SfM/MVS 파이프라인 전체를, 캘리브레이션도 포즈도 없이 이미지 쌍에서 pointmap을 직접 회귀하는 트랜스포머 하나로 갈아치운 논문 — 이후 MASt3R·모노큘러/멀티뷰 통합 계열의 출발점*

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud, *DUSt3R: Geometric 3D Vision Made Easy*, CVPR 2024. Aalto University + Naver Labs Europe.
([arXiv:2312.14132](https://arxiv.org/abs/2312.14132) · [프로젝트](https://dust3r.europe.naverlabs.com) · [코드](https://github.com/naver/dust3r))

## 한 줄 요약

> 두 이미지에서 **첫 이미지 카메라 좌표계로 표현된 두 장의 pointmap** $X^{1,1}, X^{2,1}$을 confidence와 함께 직접 회귀하도록 CroCo 사전학습 트랜스포머를 8.5M 쌍으로 학습하면, intrinsic·포즈·depth·대응점·dense 재구성이 전부 그 출력의 **후처리 부산물**이 된다 — 멀티뷰 포즈(CO3Dv2 RRA@15 96.2)와 멀티뷰 depth(평균 rel 4.73, GT 포즈 쓰는 방법들까지 제침)에서 SOTA, 나머지 태스크도 전용 방법에 근접.

## 문제

MVS in the wild는 먼저 카메라 파라미터(intrinsic + extrinsic)를 요구한다. 삼각측량에 필수이기 때문이다. 그래서 현대 파이프라인은 keypoint 검출→매칭→robust estimation→SfM/BA→dense MVS라는 **minimal problem의 연쇄**가 된다. 저자들의 진단:

- 각 서브문제가 완벽히 풀리지 않고 **노이즈를 다음 단계로 전가**한다. 파이프라인 전체의 복잡도와 엔지니어링 비용이 커진다.
- 서브문제 간 **소통이 없다**. dense 재구성은 포즈를 만든 sparse 장면 지식의 도움을 받아야 자연스럽고, 그 역도 마찬가지인데 단절돼 있다.
- 핵심 단계가 잘 부러진다. SfM의 카메라 추정은 뷰 수가 적을 때, non-Lambertian 표면, 카메라 모션이 부족할 때 흔히 실패한다. "MVS는 입력 카메라 품질만큼만 좋다."

DUSt3R는 반대 극단을 취한다: 쌍(pair) 재구성을 **pointmap 회귀** 문제로 캐스팅해 투영 카메라 모델의 하드 제약 자체를 풀어버리고, 모노큘러($I^1=I^2$)와 바이노큘러를 한 정식화로 통합한다.

## 방법

### Pointmap — 표현의 선택이 절반

Pointmap $X \in \mathbb{R}^{W\times H\times 3}$: 각 픽셀 $(i,j)$에 3D 점 하나를 대응시키는 dense 2D 필드 ($I_{i,j} \leftrightarrow X_{i,j}$). GT는 depthmap $D$와 intrinsic $K$로 $X_{i,j}=K^{-1}[iD_{i,j},\, jD_{i,j},\, D_{i,j}]^\top$. 카메라 $n$의 pointmap을 카메라 $m$ 좌표계로 표현한 것을 $X^{n,m} = P_m P_n^{-1} h(X^n)$으로 쓴다($P$는 world-to-camera, $h$는 homogeneous 매핑).

네트워크 $\mathcal{F}$는 $I^1, I^2$를 받아 $X^{1,1}, X^{2,1}$과 confidence $C^{1,1}, C^{2,1}$을 출력한다. **두 pointmap 모두 첫 이미지의 카메라 좌표계**로 표현되는 것이 핵심 — 캘리브레이션도 포즈도 모른 채 두 뷰가 처음부터 공통 좌표계에 놓이고, 상대 포즈는 두 출력 사이에 암묵적으로 인코딩된다. 픽셀↔3D의 관계는 보존하면서 원근 카메라 정식화의 제약은 강제하지 않는다("pointmap이 물리적으로 그럴듯한 카메라 모델에 대응할 필요조차 없다" — 기하 제약은 데이터에서 배운다).

### 아키텍처 — Siamese ViT + cross-attention 디코더 쌍

CroCo에서 영감을 받은 구조. 가중치 공유 ViT 인코더가 두 이미지를 각각 인코딩($F^1, F^2$)한 뒤, 두 개의 트랜스포머 디코더가 블록마다 self-attention → **cross-attention(상대 브랜치의 토큰에 attend)** → MLP를 수행하며 정보를 계속 교환한다:

$$G^1_i = \mathrm{DecoderBlock}^1_i(G^1_{i-1}, G^2_{i-1}), \qquad G^2_i = \mathrm{DecoderBlock}^2_i(G^2_{i-1}, G^1_{i-1})$$

이 상시 교환이 **정렬된 pointmap을 내는 데 결정적**이다. 마지막에 브랜치별 regression head가 디코더 토큰 전체를 받아 pointmap + confidence를 출력한다. 구성: ViT-Large 인코더, ViT-Base 디코더, DPT head. CroCo v2 사전학습 가중치로 초기화 — cross-view completion pretext는 정확히 이 구조에 맞는 3D 암묵 표현을 심어주며, ablation(224-NoCroCo vs 224)에서 전 태스크 일관된 격차로 확인된다.

### 손실 — 스케일 정규화 회귀 + confidence 가중

유효 픽셀 $i \in \mathcal{D}^v$($v\in\{1,2\}$)에 대한 3D 회귀 손실:

$$\ell_{\mathrm{regr}}(v,i) = \left\| \tfrac{1}{z} X^{v,1}_i - \tfrac{1}{\bar z} \bar X^{v,1}_i \right\|$$

예측/GT의 스케일 모호성을 처리하기 위해 각각을 norm factor로 나눈다. $z = \mathrm{norm}(X^{1,1}, X^{2,1})$, $\bar z = \mathrm{norm}(\bar X^{1,1}, \bar X^{2,1})$이고,

$$\mathrm{norm}(X^1, X^2) = \frac{1}{|\mathcal{D}^1|+|\mathcal{D}^2|} \sum_{v\in\{1,2\}} \sum_{i\in\mathcal{D}^v} \|X^v_i\|$$

즉 **유효 3D 점들의 원점까지 평균 거리**로 정규화한다. 하늘·반투명체처럼 3D가 ill-defined인 픽셀, 단일 뷰에만 보이는 어려운 영역이 있으므로 픽셀별 confidence를 함께 학습한다:

$$\mathcal{L}_{\mathrm{conf}} = \sum_{v\in\{1,2\}} \sum_{i\in\mathcal{D}^v} C^{v,1}_i\, \ell_{\mathrm{regr}}(v,i) - \alpha \log C^{v,1}_i$$

$C^{v,1}_i = 1 + \exp \tilde C^{v,1}_i > 1$로 강제해 어려운 영역에서도 외삽을 하게 만들고, $-\alpha\log C$ 항이 "전부 자신 없음"으로 도망가는 것을 막는다. confidence는 **명시적 감독 없이** 이 손실만으로 학습된다.

### Pointmap에서 공짜로 나오는 것들

- **대응점**: 3D pointmap 공간에서 상호(reciprocal) nearest-neighbor — $\mathcal{M}_{1,2} = \{(i,j)\,|\, i = \mathrm{NN}(j),\ j = \mathrm{NN}(i)\}$.
- **Intrinsic**: $X^{1,1}$은 $I^1$ 좌표계이므로, 주점 중심·정방 픽셀을 가정하면 focal만 남는다: $f^*_1 = \arg\min_f \sum_{i,j} C_{i,j} \| (i',j') - f\,(X_{i,j,0}, X_{i,j,1})/X_{i,j,2} \|$ (Weiszfeld 반복해로 수 회에 수렴).
- **상대 포즈**: $X^{1,1} \leftrightarrow X^{1,2}$를 Procrustes 정렬 — $R^*, t^* = \arg\min_{\sigma,R,t} \sum_i C_i \|\sigma(RX^{1,1}_i + t) - X^{1,2}_i\|^2$ (폐형해, 단 outlier에 민감) — 또는 더 강건하게 PnP-RANSAC.
- **Depth**: $X^{1,1}$의 z좌표가 곧 depthmap. $I^1=I^2$로 넣으면 모노큘러 depth.
- **절대 포즈(visual localization)**: 쿼리-DB 이미지 간 2D 대응 → DB의 GT 2D-3D로 PnP-RANSAC, 또는 상대 포즈를 GT pointmap 스케일로 보정.

### Global Alignment — 재투영 없는 3D 정렬

네트워크는 쌍만 다루므로, 다중 이미지 $\{I^1,\dots,I^N\}$은 후처리로 합친다. 시각적으로 겹치는 쌍들로 연결 그래프 $\mathcal{G}(\mathcal{V},\mathcal{E})$를 만들고(retrieval 또는 전 쌍 추론 후 평균 confidence 필터링, 쌍당 추론 ~40ms/H100), 엣지 $e=(n,m)$마다 예측된 $X^{n,e}, X^{m,e}$에 대해 월드 pointmap $\chi^n$, 쌍별 포즈 $P_e \in \mathbb{R}^{3\times4}$, 스케일 $\sigma_e>0$을 동시에 최적화한다:

$$\chi^* = \arg\min_{\chi, P, \sigma} \sum_{e\in\mathcal{E}} \sum_{v\in e} \sum_{i=1}^{HW} C^{v,e}_i \left\| \chi^v_i - \sigma_e P_e X^{v,e}_i \right\|$$

같은 엣지의 두 pointmap은 이미 같은 좌표계에 있으므로 **하나의 강체변환** $P_e$가 둘을 동시에 월드로 데려가야 한다는 제약이 걸린다. $\prod_e \sigma_e = 1$로 전체 붕괴($\sigma_e=0$)를 방지. 카메라를 원하면 $\chi^n_{i,j} := P_n^{-1}h(K_n^{-1}[iD^n_{i,j}; jD^n_{i,j}; D^n_{i,j}])$로 재파라미터화해 $\{P_n\},\{K_n\},\{D^n\}$을 직접 얻는다. BA와 결정적으로 다른 점: **2D 재투영 오차가 아니라 3D 투영 오차**를 최소화하며, 표준 gradient descent로 수백 스텝, 일반 GPU에서 수 초면 수렴한다.

### 학습 데이터·해상도

8개 데이터셋 혼합 총 **8.5M 쌍**: Habitat 1000k, ARKitScenes 2040k, MegaDepth 1761k, Static Scenes 3D 337k, BlendedMVS 1062k, ScanNet++ 224k, CO3Dv2 941k, Waymo 1100k — 실내/실외/합성/실사/객체중심 혼합. 쌍이 없는 데이터셋은 retrieval+매칭으로 추출. 224×224로 먼저 학습(linear head, 50 epoch) 후 최대변 512의 다양한 종횡비(512×384, 512×336, 512×288, 512×256, 512×160)로 이어 학습, 마지막에 DPT head 교체 학습. AdamW, lr 1e-4. 데이터 증강은 color jitter + 주점 중심을 유지하는 random centered crop(focal 증강 효과).

## 결과 — 하나의 모델로 태스크 5개, 정직한 위치

모든 결과가 **파인튜닝 없는 동일한 DUSt3R 512 모델**이다.

**멀티뷰 포즈 (CO3Dv2 / RealEstate10K, 10프레임)** — 가장 강한 결과. DUSt3R-GA가 RRA@15 **96.2** / RTA@15 **86.8** / mAA(30) **76.7**로 PoseDiffusion(80.5/79.8/66.5)을 크게 앞선다. RealEstate10K mAA(30) **67.7** vs PoseDiffusion 48.0(RE10K 학습판 ~80 — 단 DUSt3R는 RE10K 데이터를 전혀 안 씀). GA 없이 PnP만으로도 94.3/88.4/77.2. 3프레임만 줘도 95.3/88.3/77.5로 거의 안 떨어진다(부록 Tab.5) — 거의 반대편(~180°) 시점 쌍도 처리한다.

**멀티뷰 depth (KITTI/ScanNet/ETH3D/DTU/T&T)** — GT 포즈·depth range 없이 평균 rel **4.73** / τ(1.03) **64.52**, 프레임당 0.13s. ETH3D rel 2.91로 SOTA, GT 포즈를 쓰는 최상위 방법들(COLMAP 평균 9.3/67.8, ≈3분)보다 평균 rel에서 앞선다.

**모노큘러 depth (zero-shot)** — NYUv2 rel 6.50 / δ1.25 94.09, KITTI 10.74/86.60, BONN 8.08/93.56. self-supervised 계열은 능가하고, 지도학습 SOTA(NYUv2 DPT 5.40/96.54)에 근접.

**Visual localization (7Scenes / Cambridge)** — 2D-2D 매처로 쓰면 median 오차가 Chess 3cm/0.97°, Cambridge K.College 11cm/0.20°, St.Mary's 7cm/0.24° 수준으로 HLoc 등 feature-matching 강자와 비등(일부 장면은 우세, Stairs 11cm/2.84°는 열세). visual localization 용도로 학습된 적도, 해당 장면을 본 적도 없다는 점이 포인트.

**DTU MVS (mm 단위 재구성)** — 정직하게 밀리는 곳. Acc 2.677 / Comp 0.805 / Overall **1.741mm** vs 도메인 특화+GT 카메라 방법들(GeoMVSNet overall 0.295mm, Gipuma Acc 0.283mm). 서브픽셀 삼각측량 대신 회귀라서 정밀도 한계가 있다는 것을 저자들도 인정 — 대신 "카메라 사전지식 0에서 나온 2.7mm"라는 plug-and-play 가치를 주장한다.

## 내 실습 연결

석사에서 wide-baseline 대응점 문제를 SuperGlue/DKM/LoFTR로 후보를 만들고 guided mask로 거른 뒤 COTR로 정밀화하는 파이프라인(29.1→2.23px)으로 풀었고, 리콘랩스에서는 COLMAP→3DGS 프로덕션을 돌렸다. 그 두 경험의 공통 골격이 정확히 이 논문이 겨눈 "매칭→검증→삼각측량" 연쇄다.

- **연쇄의 흡수**: 내 매칭 파이프라인은 각 단계의 오류를 다음 단계 필터로 막는 구조였다. DUSt3R는 그 연쇄 전체를 pointmap 회귀 하나로 흡수한다 — 대응점은 출력의 NN 검색 부산물이 되고, 내가 guided mask로 하던 outlier 억제는 confidence 학습이 대신한다. "대응을 찾고 나서 3D를 세우는" 순서가 "3D를 회귀하고 나서 대응을 읽어내는" 순서로 뒤집힌 것. 매칭 연구자 입장에서 이것은 매칭의 종말이 아니라 매칭의 정의 변경이다(실제로 후속작 MASt3R가 이 구조에 매칭 head를 다시 붙인다).
- **COLMAP 운영자 관점**: 프로덕션에서 COLMAP이 부러지는 지점 — 뷰 부족, 저텍스처, non-Lambertian, 모션 부족 — 이 논문 서론의 실패 목록과 정확히 겹친다. 저텍스처에서 학습 prior가 shape-from-texture 대신 장면 통계로 메꿔주는 것은 실무적으로 매력적이다. 단, 3DGS 입력용으로 보면 DTU 1.741mm vs 0.295mm 격차가 말해주듯 기하 정밀도는 COLMAP 서브픽셀 삼각측량이 여전히 위다. prior는 "그럴듯한" 표면을 내지 "정확한" 표면을 보장하지 않으므로, 텍스처가 충분하면 고전 파이프라인, 부족하면 DUSt3R식 prior라는 사용 구분이 현재의 정직한 답으로 보인다.
- 카메라 프레임 기준 pointmap + Procrustes/PnP로 포즈를 뽑는 흐름은 내 camera-to-robot guidance 데모의 "3D perception → coordinate transform → action-ready target" 축과 같은 문법이다. 캘리브 없는 eye-in-hand 초기 정렬 같은 곳에 zero-shot으로 꽂아볼 여지가 있다.

## 한 줄 평 / 한계

> 표현(공통 좌표계 pointmap) 하나를 잘 고르면 파이프라인이 후처리로 붕괴한다는 것을 보여준 논문 — 정밀도로 이긴 게 아니라 **전제 조건을 지워서** 이겼다.

- 출력이 **스케일 불명**(metric 아님)이고, 회귀 특성상 서브픽셀 삼각측량 정밀도(DTU)에 못 미친다.
- Global alignment는 BA가 아니라 3D 오차 최소화라 빠르지만, 쌍 단위 네트워크 + $O(N^2)$ 쌍 그래프 구조는 대규모 이미지 컬렉션에서 병목이 된다(후속 연구들이 공격한 지점).
- 같은 카메라로 찍은 시퀀스에도 프레임별 intrinsic을 독립 최적화한다 — 일관성 제약을 안 쓰는 것은 유연함이자 낭비다.
- in-the-wild visual localization에서 DB pointmap이 sparse하면 스케일 복원이 무너진다(부록 E, Cambridge에서 오차 급증). 8.5M 쌍의 GT 자체가 SfM/센서 산물이라는 순환도 남는다.
