# Efficient LoFTR (CVPR 2024)

> "전체 feature map에 attention을 거는 것은 낭비다 — 토큰을 집계하고, 정제는 두 번에 나눠라" — LoFTR을 재설계해 sparse 매처급 속도를 얻은 semi-dense 매칭.

- Yifan Wang*, Xingyi He*, Sida Peng, Dongli Tan, Xiaowei Zhou (Zhejiang Univ., State Key Lab of CAD&CG), CVPR 2024
- arXiv: https://arxiv.org/abs/2403.04765
- 프로젝트: https://zju3dv.github.io/efficientloftr/

## 한 줄 요약
> LoFTR의 설계를 전면 재검토해 (1) coarse attention을 depthwise conv/max-pool로 집계한 축소 토큰 위 vanilla attention + 2D RoPE로 바꾸고, (2) fine 정제를 pixel-level MNN → 3×3 윈도 expectation의 two-stage correlation으로 재설계해, LoFTR 대비 정확도를 올리면서 ~2.5배 빠르고(optimized 모델은 SP+LightGlue보다 빠름) ROMA 대비 ~7.5배 빠른 semi-dense 매처.

## 문제
LoFTR은 detector-free semi-dense 매칭으로 저텍스처·대시점변화에 강하지만 **느리다**. 원인 진단이 이 논문의 출발점이다.

1. **Coarse attention의 낭비**: LoFTR은 1/8 해상도 feature map의 **모든 토큰**에 Transformer를 돌리고, 토큰 수가 많아 vanilla attention 대신 linear attention(φ(·)=elu(·)+1 kernel)으로 후퇴한다. 그런데 저자들의 관찰은 (a) 이웃 query 토큰들의 attention 영역이 비슷해 지역 정보가 공유·중복되고, (b) 각 query의 attention weight가 소수의 salient key 토큰에 집중된다는 것. 즉 전체 map에 대한 전역 attention 자체가 redundant하며, linear attention은 표현력 저하까지 동반한다. QuadTree attention 같은 후속작은 계층적 span 축소로 계산량은 줄이지만 attention을 여러 단계로 쪼개 **latency는 오히려 증가**한다.
2. **Fine 정제의 spatial variance**: LoFTR은 I_A의 중심 패치 feature를 고정 기준으로 두고 상대 패치 **전체**에 대한 correlation의 expectation으로 서브픽셀 위치를 구한다. correlation에 노이즈가 있으면 무관한 영역의 weight가 결과를 끌어당겨 **위치 분산(location variance)**이 생기고 이는 정확도를 깎는다.
3. **dual-softmax 비용**: coarse 매칭의 dual-softmax는 고해상도에서 토큰 수가 커질수록 추론 병목이 된다.

## 방법
파이프라인 4단계: (1) 백본 feature 추출 → (2) aggregated attention으로 coarse 변환 → (3) coarse 매칭(MNN) → (4) two-stage fine 매칭.

### 1. 백본: ResNet-FPN → 재매개변수화 RepVGG
LoFTR의 multi-branch ResNet-18+FPN 대신 **RepVGG**를 채택. 학습 시엔 residual 연결이 있는 multi-branch로 표현력을 확보하고, 추론 시엔 병렬 conv kernel을 하나로 융합(reparameterization)해 **단일 branch 네트워크로 무손실 변환**한다. 4-stage 구조로 첫 stage는 width 64/stride 1, 이후 세 stage는 width [64, 128, 256]/stride 2, 각 stage는 [1, 2, 4, 14]개의 RepVGG block. 마지막 stage의 **1/8 해상도** 출력이 coarse feature, 2·3번째 stage의 **1/2, 1/4 해상도** 출력이 fine fusion용이다.

### 2. Aggregated Attention (핵심 기여 1)
self/cross attention을 N=4번 interleave하되, 각 attention **앞에서 토큰을 지역 집계**한다:

```
f_i' = Conv2D(f_i),   f_j' = MaxPool(f_j)
```

- query 쪽(f_i)은 stride s의 **depthwise convolution**(kernel s×s), key/value 쪽(f_j)은 s×s **max-pooling**. 기본값 s=4.
- 양쪽 토큰 수가 각각 **s² = 16배** 감소 → attention 비용은 토큰 수 제곱에 비례하므로 크게 절감되고, 그 덕에 linear attention이 아닌 **vanilla attention** `softmax(QK^T)V`를 되살려 표현력을 회복한다.
- query에 pooling이 아닌 conv를 쓰는 이유(부록 A): salient 토큰이 이웃을 대표해버리면 저텍스처 영역의 attention이 인접 salient 토큰에 지배되어 성능이 떨어진다(부록 Tab.10 ablation으로 확인).
- attention 후 feature를 **upsample해 원본 f_i와 융합**(FFN 앞에서 upsample — 순서를 바꾸면 성능 하락, Tab.10).

**Positional encoding — 2D RoPE**: LoFTR의 absolute sinusoidal 대신 **2D Rotary Position Embedding**을 채택. attention score를

```
a_ij = q_i^T R(x_j − x_i, y_j − y_i) k_j
```

로 계산하며, R은 채널을 4개씩 묶어 x/y 방향 회전을 적용하는 block-diagonal 행렬, 주파수는 θ_k = 1/10000^{4k/d}. 상대 좌표만 쓰므로 특정 절대 위치가 아닌 **feature 간 상호작용**에 집중하고 회전·평행이동·스케일에 더 강건하다. **self-attention에만 적용하고 cross-attention에서는 생략**한다. RoPE vs sinusoidal ablation: AUC@5° 56.4 vs 55.5 (시간 139.2 vs 137.5ms) — 거의 공짜로 +0.9.

### 3. Coarse 매칭과 efficiency-optimized 추론
변환된 coarse feature를 dense correlation해 score matrix S를 만들고 dual-softmax + threshold τ + MNN으로 coarse 매칭 {M_c}를 뽑는다. **핵심 관찰**: dual-softmax는 판별력 있는 feature를 *학습*시키는 데 필수지만, 학습이 끝나면 **추론 시 생략하고 S에 직접 MNN을 걸어도 잘 작동**하며 더 빠르다. 이것이 **efficiency optimized 모델**(dual-softmax 생략 + Mixed-Precision)이다.

### 4. Two-Stage Correlation Refinement (핵심 기여 2)
fine feature는 별도 attention 네트워크 없이, **이미 변환된 coarse feature를 1/4·1/2 백본 feature와 conv+upsample로 융합**해 원본 해상도로 얻는다(경량 fusion만 사용). 각 coarse 매칭 중심으로 패치를 crop한 뒤:

- **1단계 (pixel-level MNN)**: fine 패치 쌍을 densely correlate해 local score matrix S_l을 만들고 **MNN 검색**으로 pixel-level 매칭을 얻는다(coarse 매칭당 top-1만 유지). MNN은 최고 score 픽셀을 직접 인덱싱하므로 **spatial variance가 없다** — 대신 서브픽셀 정확도는 없다.
- **2단계 (서브픽셀 expectation)**: 1단계에서 위치가 이미 정확해졌으므로, I_B의 fine 매칭 중심 **3×3** feature 패치만 I_A의 해당 점 feature와 correlate → softmax → **expectation**. 극소 윈도라 무관 영역의 weight 개입이 최대로 억제된 채 서브픽셀 좌표를 얻는다.

즉 "variance 없는 MNN"과 "서브픽셀 가능한 expectation"의 장점을 순서대로 결합해 LoFTR의 단일 expectation 방식의 분산 문제를 해결했다.

### 학습
loss는 L = L_c + αL_f1 + βL_f2 (α=1.0, β=0.25). L_c는 GT coarse 매칭 위치에서 S의 log-likelihood, L_f1은 S_l의 pixel-level log-likelihood, L_f2는 최종 서브픽셀 매칭의 ℓ2. MegaDepth로 end-to-end 학습, AdamW lr 4×10⁻³, batch 16, V100 8장 ~15시간. 이 단일 모델로 모든 데이터셋을 평가(cross-dataset 일반화 포함).

## 결과 (모든 수치는 원문 표, 시간은 RTX 3090)

**Relative pose AUC@5°/10°/20° (Tab.1, MegaDepth 학습 모델, 640×480 기준 시간)**

| 방법 | MegaDepth | ScanNet | 시간(ms) |
|---|---|---|---|
| SP+LightGlue | 49.9 / 67.0 / 80.1 | 14.8 / 30.8 / 47.5 | 31.9 / 30.7 |
| LoFTR | 52.8 / 69.2 / 81.2 | 16.9 / 33.6 / 50.6 | 66.2 |
| AspanFormer | 55.3 / 71.5 / 83.1 | 19.6 / 37.7 / 54.4 | 81.6 |
| **Ours** | **56.4 / 72.2 / 83.5** | 19.2 / 37.0 / 53.6 | 40.1 / 34.4 |
| **Ours (Optimized)** | 55.4 / 71.4 / 82.9 | 17.4 / 34.4 / 51.2 | **35.6 / 27.0** |
| ROMA (dense) | 62.6 / 76.7 / 86.3 | 28.9 / 50.4 / 68.3 | 302.7 |

- **속도**: LoFTR 66.2ms → optimized 27.0ms로 **~2.5배**, sparse 파이프라인 SP+LG(30.7ms)보다도 빠름. ROMA 대비 **~7.5배** 빠름(ROMA의 ScanNet 강세는 DINOv2 사전학습의 실내 데이터 덕으로 논문이 해석). AspanFormer 대비 ~2배 빠르면서 MegaDepth 전 지표 우위.
- **HPatches homography AUC@3/5/10px (Tab.2)**: Ours 66.5/76.4/85.5 vs LoFTR 65.9/75.6/84.6, SP+SG 53.9/68.3/81.7 (semi-dense는 top-1000 매칭만 사용하는 공정 조건).
- **InLoc (Tab.3)**: DUC1 52.0/74.7/86.9, DUC2 58.0/**80.9**/**89.3** (LoFTR DUC2 54.2/74.8/85.5).
- **Aachen v1.1 (Tab.4)**: Day 89.6/**96.2**/99.0, Night 77.0/91.1/99.5 — 최상위권과 동급.
- **Ablation (Tab.5, MegaDepth 1200×1200)**: Full 56.4/72.2/83.5 @139.2ms. dual-softmax 제거 → **102.0ms**(고해상도에서 효율 이득 큼). aggregated attention을 LoFTR transformer(RoPE 장착, 공정 비교)로 교체 → 54.7/70.5/82.2 @171.4ms (느려지고 부정확). two-stage 정제를 LoFTR 정제로 교체 → 54.7/70.9/82.7. 2단계 정제 제거 → 55.8/71.8/83.3 (AUC@5° −0.6). RepVGG→ResNet → 55.4/71.4/82.9 @156.2ms.
- **집계 범위 (Tab.7)**: s=2로 줄이면 56.2/72.2/83.6로 정확도는 비슷하지만 271.1ms로 **약 2배 느려짐** — s=4가 sweet spot.
- **단계별 시간 (Tab.12, 640×480)**: 총 40.1→27.0ms. dual-softmax 생략으로 coarse 매칭 8.3→1.7ms, Mixed-Precision으로 백본 9.1→5.8ms.
- **매칭 latency (Tab.13, HPatches)**: Ours 45.9/38.6ms(997 matches) vs LoFTR 76.2ms(995 matches) — 비슷한 매칭 수에서 매칭·RANSAC 시간 모두 절감.

## 내 실습 연결
석사 계측 파이프라인에서 LoFTR을 앙상블 일원으로 실사용했다(재투영 오차 29.1→2.23px 개선 과정의 매처 축). 그때 체감한 semi-dense의 가치는 명확했다 — 도장·저텍스처 표면에서 detector 기반이 후보점 자체를 못 잡을 때 LoFTR은 대응을 만들어냈다. 문제는 처리량이었고, 이 논문은 정확히 그 지점을 친다. "LoFTR보다 정확하면서 2.5배 빠르고 SP+LG보다도 빠르다"는 것은 semi-dense를 정확도 특화 옵션이 아니라 **기본 매처로 재선택**할 수 있다는 뜻이다. 특히 (1) dual-softmax를 추론에서 빼는 트릭은 내 파이프라인처럼 고해상도 입력에서 이득이 크고(139.2→102.0ms), (2) two-stage 정제의 "MNN으로 분산 제거 후 3×3 expectation" 설계는 계측처럼 서브픽셀 정확도가 곧 성능인 문제에서 LoFTR 대비 구조적으로 유리하다. 로보틱스 쪽에서도 visual localization/SLAM의 latency 제약 아래 저텍스처 강건성을 유지하는 매처로 우선 검토 대상.

## 한 줄 평 / 한계
새 패러다임 없이 기존 시스템의 병목을 정확히 진단하고 하나씩 제거한 모범적 "재설계 논문" — attention 중복성 관찰(집계), 학습/추론 요구사항 분리(dual-softmax 생략), variance 원인 분리(two-stage) 모두 진단이 처방보다 먼저다. 한계는 원문 스스로 인정하듯 (1) 강한 반복 구조(같은 의자가 있는 다른 장면)에서 실패 가능 — local feature 집중 설계라 global semantic context가 부족하고, (2) LightGlue의 early-stop 같은 적응형 계산은 미적용(orthogonal하므로 결합 여지), (3) dense ROMA와의 정확도 격차(MegaDepth AUC@5° 56.4 vs 62.6)는 여전해 속도-정확도 스펙트럼의 중간을 차지할 뿐 상한을 올리진 않는다.
