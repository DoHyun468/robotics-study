# COTR (ICCV 2021)

*대응점 찾기를 "좌표를 넣으면 좌표가 나오는 함수"로 바꾸다 — sparse와 dense를 하나의 질의 함수로*

Jiang, Trulls, Hosang, Tagliasacchi, Yi, *COTR: Correspondence Transformer for Matching Across Images*, ICCV 2021 (arXiv 2103.14167). UBC + Google Research + Toronto. 트랜스포머를 이미지 대응(correspondence) 문제에 적용한 첫 사례를 자처하는 논문이자, 내 석사 연구의 최종 정밀 매칭기로 쓴 모델.

## 한 줄 요약

> 대응점 찾기를 keypoint 검출·매칭 파이프라인이 아니라 **질의 함수 x' = F_Φ(x | I, I')** 로 정식화 — 이미지 쌍을 조건으로 받고 쿼리 좌표 x를 입력하면 상대 이미지의 대응 좌표 x'를 회귀하는 CNN+transformer를 학습하고, 추론 시 **재귀적 zoom-in**으로 정밀도를 끌어올린다. 원하는 점만 질의하면 sparse, 전 픽셀을 질의하면 dense — 하나의 모델로 양쪽을 재학습 없이 커버.

## 문제

이미지 간 대응 찾기는 전통적으로 두 갈래로 갈라져 있었다.

- **Sparse 계열** (DoG/SIFT → LIFT → SuperGlue): keypoint 검출 → 기술자(descriptor) → 매칭 → RANSAC류 robust matcher의 다단 파이프라인. wide-baseline의 큰 카메라 모션에 강하지만, **검출기가 점을 찍어준 곳에서만** 대응을 얻을 수 있다 — 임의 위치의 대응은 원리적으로 불가능하다.
- **Dense 계열** (Lucas-Kanade, PWC-Net, GLU-Net, RAFT): optical flow처럼 전 픽셀의 흐름을 추정. local smoothness에 기대므로 텍스처 없는 곳도 채우지만, 작은 시간 변화(연속 프레임)를 가정해 **큰 baseline·큰 외형 변화에서 무너진다**.

COTR의 문제의식: 이 분단은 방법론이 만든 인공적 경계다. global prior(넓은 baseline을 견디는 기하 문맥)와 local prior(smoothness)를 **한 네트워크가 데이터에서 암묵적으로 배우게** 할 수 없나?

답으로 attention이 "어느 위치를 어느 위치와 관련지을지"를 스스로 정하는 transformer를 꺼내 든다. GT flow는 물체 경계에서 불연속인데, 순수 MLP류 매끄러운 함수 근사기는 이 불연속을 표현하기 어렵고 transformer는 가능하다는 것이 핵심 직관이다.

## 방법

### 정식화 — 대응은 좌표 질의 함수다

쿼리 좌표 x ∈ [0,1]² (이미지 I의 정규화 좌표)에 대해 대응점 x' ∈ [0,1]² (이미지 I')를 내는 파라메트릭 함수 F_Φ(x | I, I')를 학습한다. 목적식은 두 항의 기대값 최소화(argmin_Φ E[L_corr + L_cycle]):

- **대응 손실**: L_corr = ‖x' − F_Φ(x | I, I')‖²₂ — GT 대응과의 좌표 회귀 오차.
- **cycle consistency 손실**: L_cycle = ‖x − F_Φ(F_Φ(x | I, I') | I', I)‖²₂ — 갔다가 돌아오면 제자리여야 한다. 학습 정칙화이자, 추론에서는 신뢰도 필터로 재활용된다(아래).

이 함수형 관점이 논문의 본체다. 검출기가 없으므로 **어느 좌표든 질의할 수 있고**, 질의 수를 조절해 sparse↔dense를 자유롭게 오간다. DeepSDF가 "형상 = 좌표→SDF 함수"로 본 것의 대응 문제 버전이라 할 수 있다.

### 아키텍처 — CNN 백본 + transformer encoder-decoder

- 입력 두 이미지를 각각 256×256으로 리사이즈, 공유 CNN 백본 E(ImageNet 사전학습 ResNet50, 3번째 residual block 뒤 16×16×1024 특징을 1×1 conv로 256채널화)로 16×16×256 특징맵을 만든다.
- 두 특징맵을 **채널이 아니라 공간 방향으로 나란히 연접**해 16×32×256을 만들고, 좌표 함수 Ω(16×32×2 MeshGrid(0:1, 0:2))의 위치 인코딩을 더해 context 특징맵을 얻는다: c = [E(I), E(I')] + P(Ω).
- 공간 연접이 핵심 설계다. 채널 연접은 같은 픽셀 위치의 특징끼리 인위적 관계를 만들지만, 공간 연접은 두 이미지의 위치들을 문장의 단어들처럼 다뤄 encoder T_E가 self-attention(이미지 내)과 cross-attention(이미지 간)을 한 attention 안에서 자연스럽게 배우게 한다. 연접 맵 전체에 단일 위치 인코딩을 써서 두 이미지의 픽셀이 구분된다.
- **선형 증가 주파수 위치 인코딩**: 위치 x = [x, y]에 대해 P(x) = [p₁(x), …, p_{N/4}(x)], p_k(x) = [sin(kπxᵀ), cos(kπxᵀ)], N=256 (p_k가 4개 값을 내므로 출력이 정확히 N차원). 통상의 log-linear 주파수 스케줄은 최적화가 불안정했고, **주파수를 k에 선형으로** 올린 것이 결정적이었다고 명시한다.
- context 특징맵 c를 transformer encoder T_E에 넣고, 쿼리 좌표 x를 같은 인코더 P로 임베딩해 decoder T_D의 쿼리로 사용, 마지막에 3층 MLP D(256 유닛, ReLU)가 좌표를 출력한다: x' = D(T_D(P(x), T_E(c))).
- encoder/decoder 각 6층, attention 8-head. **decoder에는 쿼리 간 self-attention이 없다** — 여러 점을 한 번에 질의해도 각 쿼리가 독립적으로 풀리도록 의도적으로 막았다.

### 추론 — 재귀적 zoom-in

transformer attention의 대가로 특징맵이 16×16까지 다운샘플되어 있어 한 번의 추정은 부정확하다. 함수형이라는 성질을 이용해 **같은 네트워크를 재귀 적용**한다: 이전 추정 대응 주변을 두 이미지 모두에서 잘라(crop) 다시 256×256으로 넣고 정제 — 검증 데이터 ablation으로 **zoom 배율 2씩, 4회 zoom-in**이 계산량-정확도 절충으로 채택됐다.

- **스케일 보정**: 첫 단계에서 전체 이미지의 픽셀별 cycle consistency 오차를 τ_visible=5px(256×256 기준)로 threshold해 **co-visible 영역**을 추출하고, 이후 단계의 crop 크기를 두 이미지의 유효 영역 합에 비례시켜 스케일 불일치를 보정한다.
- **오류 기각**: cycle consistency 오차가 τ_cycle=5px를 넘는 대응, zoom-in 추정치의 표준편차가 이미지 긴 변의 τ_std=0.02를 넘는(수렴 안 하는) 대응은 버린다 — occlusion·화면 밖 질의를 걸러내는 장치.
- **dense 보간**: 전 픽셀 질의 대신 sparse 매칭 후 쿼리들의 Delaunay 삼각분할 위 barycentric 가중 보간으로 densify(GPU rasterizer로 효율화) — 'COTR+Interp.' 변형.
- **임의 크기 이미지**: 첫 단계는 그냥 256×256으로 stretch, 이후 zoom부터 원본에서 정사각 patch를 잘라 쓴다. 극단적 종횡비(예: 2:1)는 타일 2개로 나눠 cycle consistency가 가장 좋은 추정을 고른다.

### 학습

- **데이터**: MegaDepth(SfM depth가 붙은 photo-tourism, 학습 115 scene + 검증 1 scene). 3D 공통점 없는 쌍을 거른 뒤 투영 겹침(intersection over union)을 계산, 이미지당 겹침 큰 20쌍을 유지.
- **on-the-fly 생성**: 랜덤 쿼리점을 찍고 GT depth로 대응을 구한 뒤, log scale 균일 10단계(1×~10×) 랜덤 zoom crop 쌍에서 유효 대응 100개를 샘플링(100개 못 모으면 그 쌍은 폐기).
- **3단계 학습**: (1) 백본 동결, ADAM lr 10⁻⁴, batch 24로 300k iter → (2) 전체 미세조정 lr 10⁻⁵, batch 16로 2M iter(검증 손실 정체까지) → (3) zoom-in crop 도입 후 추가 300k iter. 1·2단계는 256×256 전체 이미지만 써서 데이터셋 전체를 메모리에 올린다.

## 결과

전 벤치마크에서 **재학습·미세조정 일절 없이** 같은 모델 하나로 평가한 것이 강조점.

**HPatches** (homography, dense 평가) — AEPE와 PCK:

| 방법 | AEPE↓ | PCK-1px↑ | PCK-3px↑ | PCK-5px↑ |
|---|---|---|---|---|
| DGC-Net | 33.26 | 12.00 | – | 58.06 |
| GLU-Net | 25.05 | 39.55 | 71.52 | 78.54 |
| GLU-Net+GOCor | 20.16 | **41.55** | – | 81.43 |
| **COTR** | **7.75** | 40.91 | **82.37** | **91.10** |
| COTR+Interp. | 7.98 | 33.08 | 77.09 | 86.33 |

**KITTI** (optical flow) — AEPE / Fl(outlier 비율 %):

| 방법 | 2012 AEPE↓ | 2012 Fl↓ | 2015 AEPE↓ | 2015 Fl↓ |
|---|---|---|---|---|
| GLU-Net | 3.34 | 18.93 | 9.79 | 37.52 |
| RAFT | 2.15 | 9.30 | 5.04 | 17.8 |
| GLU-Net+GOCor | 2.68 | 15.43 | 6.68 | 27.57 |
| **COTR** | **1.28** | **7.36** | **2.62** | **9.92** |
| COTR+Interp. | 2.26 | 10.50 | 6.12 | 16.90 |

단, COTR 행은 cycle consistency 필터를 통과한 신뢰 질의(전체의 81.8%)에서만 평가한 수치임을 논문 스스로 명시한다(기각분의 67.8%는 상대 이미지 경계 밖). 보간판은 필터 없이 RAFT와 대등. MegaDepth의 정적 rigid 장면(건물 파사드)만 보고 배웠는데 KITTI의 서로 반대 방향으로 움직이는 차들을 각각 잡아낸다 — global motion에 편향된 GLU-Net과의 질적 차이로, 진짜 local 대응을 배웠다는 증거.

**ETH3D** (프레임 간격 rate를 늘려 baseline 확대, AEPE): rate=3에서 COTR 1.66으로 LiteFlowNet과 동률, rate가 커질수록 격차가 벌어져 **rate=15에서 COTR 2.61** vs LiteFlowNet 74.96, PWC-Net 43.41, RAFT 13.74, GLU-Net 10.78. baseline이 넓어질수록 강해지는 dense 방법이라는 주장을 가장 선명하게 보여주는 표다.

**IMC2020** (wide-baseline stereo pose, DEGENSAC 결합): N=2048 매치에서 mAA(5°) 0.444 / mAA(10°) 0.580으로 총 228 엔트리 중 **종합 2위**(semantic masking 등 챌린지 특화 휴리스틱 엔트리 제외 시 8k 부문 포함 최상위권). 2k-keypoint 부문 우승자 SuperGlue(0.416/0.552)를 **512 매치만으로도** 능가(0.418/0.555), 준우승 DISK는 256 매치로 능가. 챌린지 특화 튜닝 없이, keypoint 개념도 없이 **랜덤 위치 질의**로 낸 수치다.

**Ablation**:

- transformer를 MLP로 바꾸면 전역적으로 매끄러운 warp만 내놓아 3D 구조가 만드는 불연속을 못 잡는다(평면적 warp 아티팩트) — transformer 필요성의 직접 증거.
- 필터링은 ETH3D AEPE를 상대 약 5% 개선하며 평균 1.2%의 대응만 기각한다.
- zoom-in 단계가 깊어질수록 HPatches 오차 히스토그램이 뚜렷하게 왼쪽으로 이동한다.

**속도 한계**: 논문 스스로 "attention은 강력하지만 비싸다"고 인정한다. 쿼리마다 crop을 만들어 4회 재귀 추론을 도는 구조라 질의당 forward가 여러 번이고, 이미지쌍당 매치 수 상한 방식의 전통적 매칭 벤치 틀에 맞지 않아 IMC 주최 측의 배려로 평가했다는 각주까지 있다 — 실시간 응용에는 무거운 방법이다.

## 내 실습 연결

석사 연구(비전 기반 치수 계측)에서 COTR은 **최종 정밀 매칭기**였다. 계측 문제의 본질은 "검출기가 골라준 점"이 아니라 **내가 지정한 계측점**의 대응을 구하는 것 — keypoint 기반 sparse 매처로는 원리적으로 안 되는 요구인데, COTR의 함수형 정식화는 **쿼리 좌표를 계측점으로 직접 지정**할 수 있어 문제 구조와 정확히 맞았다. 이 리뷰의 '방법' 절이 곧 그 선택의 근거다.

다만 단독으로는 무너졌다: 계측 대상의 반복 패턴(유사 구조가 화면 여러 곳에 존재)에서 transformer의 전역 attention이 **다른 위치의 닮은 구조로 혼동**해 대응이 크게 튀었다 — 단독 COTR 29.1px 오차. 그래서 U-Net 기반 guided mask로 co-visible/유효 영역을 먼저 제약해 attention이 볼 문맥 자체를 좁힌 뒤 COTR로 정밀 추정하는 파이프라인을 만들어 **2.23px, PCK-15px 34.2% → 84.6%** 로 끌어올렸다.

흥미로운 대칭: COTR 자신도 cycle consistency로 co-visible 영역을 추정해 스케일을 보정한다(3.3절). 내 파이프라인은 그 "영역 제약" 아이디어를 학습된 세그멘테이션으로 앞단에 옮겨, 전역 문맥 혼동이라는 COTR의 약점을 COTR의 설계 사상으로 고친 셈이다.

## 한 줄 평 / 한계

**한 줄 평.** "대응 = 좌표 질의 함수"라는 정식화 하나로 sparse/dense 분단을 허문 우아한 논문 — 검출기 없이 임의 점을 질의한다는 성질은 계측·robot guidance처럼 "**어느 점의 대응이 필요한지 문제가 먼저 정해주는**" 응용에서 특히 빛난다. 재귀 zoom-in도 함수형이라서 거의 공짜로 얻는 정제 절차라는 점이 깔끔하다.

**한계.**

- **느리다**: 쿼리당 4회 재귀 forward — dense는 Delaunay 보간으로 때우지만, 그 보간이 KITTI에서 RAFT 대비 열세의 원인으로 논문 스스로 지목된다("improved interpolation strategies based on CNNs" 를 future work로 남김). 실시간·대량 매칭에는 부적합.
- **평가의 조건부성**: KITTI 수치는 신뢰 질의(81.8%)에 대한 것으로 완전 dense 방법과 직접 비교가 어렵다 — 논문이 정직하게 밝히지만 표만 보면 놓치기 쉽다.
- **전역 attention의 양날**: 넓은 baseline을 견디는 힘이 곧 반복 구조에서의 혼동 원인이 된다 — 내 실측(단독 29.1px)이 보여준 실패 모드로, 문맥을 제약해 줄 앞단(마스크·co-visibility)과 결합할 때 진가가 나온다.
- **고정 256×256 입력·16×16 특징맵**: 해상도 병목이 구조적이라 정밀도가 전적으로 zoom-in 재귀에 의존하고, 극단적 종횡비는 타일링으로 우회해야 한다.
