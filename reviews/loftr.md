# LoFTR (CVPR 2021)

> "코너를 검출하지 말고, 전 픽셀을 Transformer로 매칭하라" — detector-free + coarse-to-fine 매칭의 표준을 세운 논문.

- Jiaming Sun*, Zehong Shen*, Yuang Wang*, Hujun Bao, Xiaowei Zhou (Zhejiang Univ. / SenseTime Research), CVPR 2021
- arXiv: https://arxiv.org/abs/2104.00680
- 프로젝트: https://zju3dv.github.io/loftr/

## 한 줄 요약
> feature detection–description–matching 3단계를 버리고, CNN dense feature에 Linear Transformer self/cross attention을 얹어 1/8 해상도에서 dual-softmax로 coarse 매칭을 만들고 1/2 해상도 윈도에서 expectation으로 서브픽셀 정제하는 detector-free 매처 — 저텍스처/반복패턴 영역에서 detector 기반(SuperGlue)을 큰 폭으로 이긴다.

## 문제
전통 파이프라인은 (1) interest point 검출 → (2) descriptor 추출 → (3) NN/학습 기반 매칭 순서다. 이 구조의 근본 약점은 **detector의 반복성(repeatability)**: 저텍스처 벽/바닥, 반복 패턴, viewpoint·조명 변화, motion blur에서 두 이미지 모두에서 재검출되는 코너 자체가 부족하다. 완벽한 descriptor가 있어도 검출이 안 되면 대응이 불가능하다(논문 Fig.1: 텍스처 없는 벽에서 SuperGlue 실패, LoFTR 성공). SuperGlue조차 attention 범위가 "검출된 interest point"로 제한된다.

기존 detector-free 계열(NCNet, Sparse-NCNet, DRC-Net)은 4D cost volume + 4D convolution으로 dense 매칭을 하지만, convolution의 receptive field가 이웃 영역에 갇혀 indistinctive한 영역을 구분하지 못한다. 사람은 저텍스처 영역도 **global context**(엣지까지의 상대 위치 등)로 구분한다 — 즉 global receptive field가 핵심이라는 관찰이 출발점이다.

## 방법
전체 구조 4단계 (Fig.2): Local Feature CNN → Coarse-Level Transformer → Matching Module → Coarse-to-Fine Module.

### 1. Local Feature CNN
수정된 ResNet-18 + FPN 백본으로 두 해상도의 feature를 뽑는다.
- coarse: F̃^A, F̃^B — 원본의 **1/8** 해상도
- fine: F̂^A, F̂^B — 원본의 **1/2** 해상도

CNN의 translation equivariance/locality가 local feature 추출에 적합하고, downsampling이 Transformer 입력 길이를 줄여 계산량을 관리 가능하게 만든다.

### 2. LoFTR Module (coarse feature transform)
coarse feature를 flatten하고 **positional encoding**(DETR식 2D sinusoidal 확장, 백본 출력에 **1회만** 더함)을 추가한 뒤, self-attention과 cross-attention 레이어를 **N_c번 interleave**한다.
- self-attention: f_i, f_j가 같은 이미지(F̃^A 또는 F̃^B)
- cross-attention: (F̃^A, F̃^B) 또는 (F̃^B, F̃^A) — 양방향

vanilla attention은
```
Attention(Q, K, V) = softmax(QK^T) V
```
로 sequence 길이 N에 대해 **O(N²)**. 1/8로 줄여도 dense 매칭에는 비현실적이라, **Linear Transformer** (Katharopoulos et al.)를 채택:
```
sim(Q, K) = φ(Q) · φ(K)^T,   φ(·) = elu(·) + 1
```
kernel trick으로 softmax(exp kernel)를 대체하면 행렬곱 결합법칙에 의해 φ(K)^T V를 먼저 계산할 수 있고, feature 차원 D ≪ N이므로 복잡도가 **O(N)**으로 떨어진다. positional encoding 덕에 흰 벽처럼 입력 RGB가 균일해도 변환된 feature는 위치마다 unique해진다(Fig.4c의 smooth color gradient).

### 3. Coarse 매칭: dual-softmax (또는 optimal transport)
변환된 feature 간 score matrix:
```
S(i, j) = (1/τ) · ⟨F̃^A_tr(i), F̃^B_tr(j)⟩
```
**dual-softmax**로 양방향 soft mutual nearest neighbor 확률을 만든다:
```
P_c(i, j) = softmax(S(i, ·))_j · softmax(S(·, j))_i
```
(OT 변형: −S를 partial assignment 비용으로 두고 SuperGlue처럼 Sinkhorn 반복.)

**Match selection**: confidence P_c > θ_c 필터 + mutual nearest neighbor(MNN) 강제 →
```
M_c = {(ĩ, j̃) | ∀(ĩ, j̃) ∈ MNN(P_c), P_c(ĩ, j̃) ≥ θ_c}
```

### 4. Coarse-to-Fine: expectation 기반 서브픽셀 정제
각 coarse match (ĩ, j̃)의 fine-level 위치 (î, ĵ)에서 **w×w** local window 두 개를 crop → 작은 LoFTR module(N_f layers)로 변환 → F̂^A_tr(î)의 **중심 벡터**를 F̂^B_tr(ĵ)의 모든 벡터와 correlate → ĵ 주변 매칭 확률 **heatmap** 생성 → 확률분포에 대한 **expectation**으로 서브픽셀 좌표 ĵ'를 얻는다. 모든 (î, ĵ')를 모아 최종 매칭 M_f.

### 손실함수
```
L = L_c + L_f
```
- **Coarse**: ground-truth pose+depth로 1/8 grid의 mutual nearest neighbor를 GT 매칭 M_c^gt로 만들고, negative log-likelihood:
```
L_c = −(1/|M_c^gt|) Σ_{(ĩ,j̃)∈M_c^gt} log P_c(ĩ, j̃)
```
- **Fine**: GT 위치 ĵ'_gt는 î를 GT pose·depth로 warp해 계산. heatmap의 total variance σ²(î)로 불확실성을 측정해 가중한 ℓ2:
```
L_f = (1/|M_f|) Σ_{(î,ĵ')∈M_f} (1/σ²(î)) · ‖ĵ' − ĵ'_gt‖₂
```
σ²(î)에는 gradient를 흘리지 않고, warp된 위치가 윈도 밖이면 해당 쌍은 무시한다.

### 하이퍼파라미터·구현 (원문 §3.6)
- 백본: modified ResNet-18, 랜덤 초기화 end-to-end 학습
- **N_c = 4, N_f = 1, θ_c = 0.2, w = 5**
- indoor: ScanNet, outdoor: MegaDepth로 별도 학습. Adam lr 1e-3, batch 64, 64대 GTX 1080Ti에서 24시간(ScanNet)
- 속도: 640×480 pair 기준 RTX 2080Ti에서 dual-softmax **116 ms**, OT(3 sinkhorn iters) **130 ms**
- fine 단계 구현: F̂_tr를 upsample한 F̃_tr와 concat해 사용

## 결과 (원문 수치)

### Homography (HPatches, corner error AUC)
| Method | @3px | @5px | @10px |
|---|---|---|---|
| SP + SuperGlue | 53.9 | 68.3 | 81.7 |
| Sparse-NCNet | 48.9 | 54.2 | 67.1 |
| DRC-Net | 50.6 | 56.2 | 68.3 |
| **LoFTR-DS** | **65.9** | **75.6** | **84.6** |

### 상대 pose — ScanNet indoor (pose AUC)
| Method | @5° | @10° | @20° |
|---|---|---|---|
| ORB + GMS | 5.21 | 13.65 | 25.36 |
| SP + SuperGlue | 16.16 | 33.81 | 51.84 |
| DRC-Net† | 7.69 | 17.93 | 30.49 |
| LoFTR-OT | 21.51 | 40.39 | **57.96** |
| **LoFTR-DS** | **22.06** | **40.8** | 57.62 |

### 상대 pose — MegaDepth outdoor (pose AUC)
| Method | @5° | @10° | @20° |
|---|---|---|---|
| SP + SuperGlue | 42.18 | 61.16 | 75.96 |
| DRC-Net | 27.01 | 42.96 | 58.31 |
| LoFTR-OT | 50.31 | 67.14 | 79.93 |
| **LoFTR-DS** | **52.8** | **69.19** | **81.18** |

저자 요약: outdoor에서 DRC-Net 대비 AUC@10° **61%**, SuperGlue 대비 **13%** 향상. indoor는 DS, outdoor도 DS가 OT보다 근소 우위.

### Visual localization
- **Aachen v1.1 night (local feature track)**: LoFTR-DS 72.8 / **88.5** / **99.0** — night에서 전 baseline 제압, day는 SP+SuperGlue와 유사하거나 약간 열세.
- **InLoc**: HLoc+LoFTR-OT가 DUC1 47.5/**72.2**/**84.8**, DUC2 **54.2**/74.8/**85.5** — texture-less·대칭·반복 요소가 많은 indoor에서 published 방법 중 1위.

### Ablation (ScanNet, OT, pose AUC @5/10/20°)
- LoFTR module을 동급 파라미터 convolution으로 교체: 14.98/32.04/49.92 — Transformer 기여가 결정적 (Full: 20.06/40.8/57.62)
- positional encoding을 DETR처럼 매 레이어 반복: 18.02/35.64/52.77 — 1회 주입이 더 낫다
- N_c=8, N_f=2로 2배 확장: 20.87/40.23/57.56 — 거의 변화 없음 (N_c=4로 충분)

## 내 실습 연결
석사에서 대형 선체블록 wide-baseline 스테레오 대응을 풀 때 SuperGlue·DKM·**LoFTR** 앙상블을 실사용했다(최종 reprojection 29.1 → 2.23 px). 이 논문을 다시 읽으니 그때 LoFTR가 앙상블에서 맡았던 역할이 명확해진다.

- 선체블록 철제 표면은 이 논문이 정확히 겨냥한 실패 모드다: 도장된 균일 표면이라 **검출할 코너 자체가 없고**, 용접선·격자 보강재는 반복 패턴이다. detector 기반(SuperGlue)은 SuperPoint가 keypoint를 못 뽑는 영역에서 구조적으로 침묵하는 반면, LoFTR는 1/8 grid 전체가 매칭 후보라 "빈 표면 한가운데"에도 대응을 깔아준다. Fig.1의 텍스처 없는 벽 사례가 우리 블록 표면과 같은 상황.
- ScanNet 실험 셋업(wide baseline + extensive texture-less)이 우리 문제 설정과 겹친다. 거기서 SuperGlue 16.16 → LoFTR 22.06 (AUC@5°)의 갭은 우리가 앙상블에서 체감한 "LoFTR가 커버리지를 채우고 SuperGlue가 고신뢰 anchor를 주는" 상보성과 일치한다.
- coarse 1/8 grid의 한계도 실무에서 체감한 그대로: LoFTR 단독 매칭은 grid에 정렬된 반면(query 쪽은 항상 1/8 grid 중심), fine 정제는 target 쪽만 서브픽셀이다. 우리가 2.23 px까지 내리는 데 DKM(dense warp)과의 앙상블 + 기하 검증이 필요했던 이유.
- 로보틱스 연결: camera-to-robot guidance에서 다루는 저텍스처 산업 부품(금속·플라스틱 몰드)도 동일한 detector 실패 영역이라, pose estimation용 대응 확보에 detector-free 계열이 기본 선택지가 된다. dual-softmax confidence P_c는 RANSAC 전 필터로 그대로 쓸 수 있다.

## 한 줄 평 / 한계
detector라는 병목을 제거하고 "global context로 위치를 유일하게 만든다"는 관찰을 Linear attention으로 실현한, 문제 정의가 깨끗한 논문 — 이후 semi-dense matching 계보(QuadTree, ASpanFormer, Efficient LoFTR)의 출발점.

한계:
- coarse match의 query 쪽은 1/8 grid에 고정 — 양쪽 서브픽셀이 아니어서 극한 정밀도(예: calibration급)에는 후처리나 dense 방법(DKM류) 보완이 필요.
- 매칭 수 상한(~1K)과 116 ms/pair는 실시간 SLAM 프론트엔드로는 무겁다.
- fine 단계 w=5 윈도는 coarse가 틀리면 복구 불가 — coarse 오매칭이 그대로 outlier가 된다.
- indoor/outdoor 별도 모델 학습이 필요하며, ablation에서 모델 확장(N_c=8)의 이득이 없다는 점은 데이터·구조의 스케일링 한계를 시사.
