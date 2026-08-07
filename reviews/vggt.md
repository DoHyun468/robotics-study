# VGGT: Visual Geometry Grounded Transformer (CVPR 2025)

*SfM 파이프라인 전체 — 카메라, depth, 포인트맵, 트랙 — 를 단 한 번의 transformer forward로. 최적화 없이 0.2초, 그런데 최적화 기반보다 정확하다*

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny (Oxford VGG + Meta AI), CVPR 2025. ([arXiv:2503.11651](https://arxiv.org/abs/2503.11651), [code](https://github.com/facebookresearch/vggt))

## 한 줄 요약

> 3D 유도 편향을 거의 뺀 1.2B 파라미터 transformer가 **1~수백 장의 이미지를 한 번의 feed-forward(1초 미만)로** 받아 카메라 내·외부 파라미터, depth map, point map, 3D 포인트 트랙을 **동시에** 출력한다 — 쌍 단위 예측 + global alignment 후처리에 묶여 있던 DUSt3R/MASt3R, 미분가능 BA를 돌리는 VGGSfM을 정확도와 속도 모두에서 넘어섰고(예: ETH3D Overall 1.005→0.677, ~7s→~0.2s), BA를 얹으면 더 오른다.

## 문제

전통 3D 재구성은 correspondence → triangulation → Bundle Adjustment의 반복 최적화(COLMAP)이고, 학습 기반이 끼어들어도 역할은 부분적이었다(keypoint, matching, 미분가능 BA를 넣은 VGGSfM까지도 geometry 최적화가 본체). DUSt3R/MASt3R가 "포인트맵 직접 회귀"로 패러다임을 바꿨지만 **한 번에 두 장**만 처리할 수 있어, N장 장면은 쌍별 예측을 global alignment로 꿰매는 후처리(장면당 ~10초)가 필수였다. 저자들의 질문은 단순하다 — 3D 태스크를 기하 후처리 없이 **네트워크 forward만으로** 풀 수 있는가, 그리고 그때 특별한 3D 아키텍처가 필요한가(답: 둘 다 아니오에 가깝다 — 거의 표준 ViT면 된다).

## 방법

### 문제 정의와 파라미터화

입력은 같은 장면을 보는 N장 $(I_i)_{i=1}^N$, 출력은 프레임별 4종:

$$f\big((I_i)_{i=1}^N\big) = (\mathbf{g}_i, D_i, P_i, T_i)_{i=1}^N$$

- **카메라** $\mathbf{g}_i \in \mathbb{R}^9 = [\mathbf{q}, \mathbf{t}, \mathbf{f}]$: 회전 쿼터니언(4) + 병진(3) + field-of-view(2). 주점은 이미지 중심 가정.
- **Depth map** $D_i \in \mathbb{R}^{H\times W}$, **point map** $P_i \in \mathbb{R}^{3\times H\times W}$. 포인트맵은 DUSt3R처럼 viewpoint invariant — 모든 3D 점이 **첫 번째 카메라 $\mathbf{g}_1$ 좌표계**(=세계 기준계)로 표현된다. 첫 카메라의 extrinsic 출력은 항등($\mathbf{q}_1=[0,0,0,1]$, $\mathbf{t}_1=\mathbf{0}$)으로 고정.
- **트래킹 피처** $T_i \in \mathbb{R}^{C\times H\times W}$: 트랙 자체가 아니라 별도 트래킹 모듈 $\mathcal{T}$의 입력이 되는 dense feature. 질의점 $\mathbf{y}_q$를 주면 전 프레임 대응 2D 점을 낸다. $f$와 $\mathcal{T}$는 end-to-end 공동 학습.

입력 순서는 임의(첫 프레임만 기준계 역할) — 아키텍처는 첫 프레임을 제외한 나머지에 대해 permutation equivariant다.

### Alternating-Attention 백본

각 이미지를 DINOv2로 패치화해 토큰 $\mathrm{t}^I$를 얻고, 프레임마다 **카메라 토큰 1개 + 레지스터 토큰 4개**를 붙인다. 첫 프레임의 카메라/레지스터 토큰만 **다른 학습 토큰**을 써서 모델이 기준 프레임을 구별하게 한다. 본체는 **frame-wise self-attention(프레임 내 토큰만) ↔ global self-attention(전 프레임 토큰 전체)을 교대**하는 Alternating-Attention(AA) $L=24$ 블록 — cross-attention은 전혀 쓰지 않는다. 프레임 내 정규화와 프레임 간 통합의 균형이 요점이고, 부록 기준 각 attention은 ViT-L급(dim 1024, 16 heads), QKNorm + LayerScale(0.01)로 안정화. 총 1.2B 파라미터.

### 예측 헤드

- **Camera head**: 출력 카메라 토큰들에 self-attention 4층 + linear를 얹어 $\hat{\mathbf{g}}_i$ 회귀.
- **Dense head**: 출력 이미지 토큰을 DPT(4·11·17·23번째 블록 토큰 사용)로 dense feature $F_i$로 업샘플하고, 3×3 conv로 $D_i$, $P_i$를 낸다. 같은 DPT가 트래킹 피처 $T_i$와 **aleatoric 불확실성 맵** $\Sigma^D_i, \Sigma^P_i$도 함께 출력.
- **Tracking head**: CoTracker2 구조. 질의점 피처를 각 프레임 $T_i$와 correlation → self-attention으로 2D 대응점 $\hat{\mathbf{y}}_i$ 예측. 시간 순서를 가정하지 않아 비디오가 아닌 무순서 이미지 집합에도 적용된다.

### Over-complete 예측 — 이 논문에서 제일 흥미로운 부분

포인트맵 $P$는 depth $D$와 카메라 $\mathbf{g}$의 닫힌 형식 함수(unprojection)라 셋을 모두 예측하는 건 중복이다. 그런데 **학습 때는 전부 명시적으로 예측시키는 게 이득**이고(multi-task ablation: 카메라 손실 제거 시 ETH3D Overall 0.709→0.834, depth 제거 시 0.727, track 제거 시 0.790), **추론 때는 전용 포인트맵 헤드보다 depth×camera unprojection이 더 정확**하다(ETH3D Overall 0.709→0.677). 복잡한 태스크를 단순 하위문제로 분해해 예측을 조합하는 쪽이 낫다는 것.

### 손실과 학습 규모

$$\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + \lambda \mathcal{L}_{\text{track}}, \quad \lambda = 0.05$$

카메라는 Huber, depth/pmap은 DUSt3R식 confidence-aware(aleatoric) 손실 $\|\Sigma\odot(\hat{D}-D)\| + \|\Sigma\odot(\nabla\hat{D}-\nabla D)\| - \alpha\log\Sigma$ 에 gradient 항 추가, 트랙은 L1 + 가시성 BCE. GT는 첫 카메라 좌표계로 옮긴 뒤 원점까지 평균 유클리드 거리로 스케일 정규화하되 — DUSt3R와 달리 — **예측에는 정규화를 걸지 않고 네트워크가 그 정규화를 배우게 강제**한다(예측측 정규화는 수렴에 불필요하고 학습 불안정만 키운다는 게 저자 관찰).

학습: Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Habitat, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria Synthetic/Digital Twin, Objaverse류 합성 데이터 등(MASt3R급 규모·다양성). AdamW 160K iter, peak LR 2e-4(cosine, warmup 8K), 배치당 장면별 2~24프레임(총 48프레임), 최대 변 518px, **A100 64장 × 9일**, bfloat16 + gradient checkpointing.

### 추론 속도와 메모리 (Tab.9, H100 + FlashAttention v3, 336×518)

| 입력 프레임 | 1 | 2 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| 시간(s) | 0.04 | 0.05 | 0.14 | 0.31 | 1.04 | 3.12 | 8.75 |
| 메모리(GB) | 1.88 | 2.07 | 3.63 | 5.58 | 11.41 | 21.15 | 40.63 |

카메라 헤드는 런타임의 ~5%, 메모리의 ~2%로 경량이고, DPT 헤드는 프레임당 평균 0.03초/0.2GB — 프레임 간 관계는 백본에서만 처리되므로 메모리가 부족하면 dense 헤드를 프레임별로 나눠 돌릴 수 있다. DUSt3R가 32장에서 이미 200초+ 및 OOM 한계인 것과 대비된다.

### 설계 노트 (부록 Discussions)

- **패치화**: 14×14 conv 대비 DINOv2 사전학습 토크나이저가 성능·학습 안정성(특히 초기)·하이퍼파라미터 민감도 모두에서 우세해 기본값으로 채택.
- **미분가능 BA**: VGGSfM식 in-the-loop BA는 소규모 실험에서 유망했으나 PyTorch(Theseus) 기준 학습 스텝이 ~4배 느려져 제외 — 3D 주석 없는 데이터의 무감독 학습 신호로는 유망하다고 남겨둔다.
- **단일 뷰**: DUSt3R처럼 이미지를 복제해 쌍을 만들 필요 없이 단일 이미지 입력이 자연스럽게 동작(global attention이 frame attention으로 퇴화). 단일뷰 학습을 따로 하지 않았는데도 유화·스케치까지 그럴듯한 포인트맵을 낸다(Fig.3, Fig.7).

## 결과 (원문 수치)

**카메라 포즈 (Tab.1, AUC@30↑, 장면당 10프레임)** — RealEstate10K는 전 방법 미학습(unseen):

| | Re10K | CO3Dv2 | 시간 |
|---|---|---|---|
| COLMAP+SPSG | 45.2 | 25.3 | ~15s |
| DUSt3R | 67.7 | 76.7 | ~7s |
| MASt3R | 76.4 | 81.8 | ~9s |
| VGGSfM v2 | 78.9 | 83.4 | ~10s |
| **VGGT (feed-forward)** | **85.3** | **88.2** | **~0.2s** |
| VGGT + BA | 93.5 | 91.8 | ~1.8s |

BA 후처리도 싸다 — 예측된 포즈/포인트가 이미 정답 근처라 triangulation·반복 정제가 필요 없어 VGGSfM(~10s)보다 훨씬 빠른 ~1.8s. 부록 IMC(phototourism)에서도 feed-forward AUC@10 71.26(0.2s)로 VGGSfMv2(76.82, ~10s)에 근접, **+BA 시 84.91로 CVPR'24 IMC 챌린지 1위였던 VGGSfMv2를 크게 추월**(AUC@3은 39.23→66.37).

**DTU 멀티뷰 depth (Tab.2, Chamfer Overall↓)**: GT 카메라 없이 DUSt3R 1.741 → **VGGT 0.382**. GT 카메라를 아는 GeoMVSNet(0.295)에 육박.

**ETH3D 포인트맵 (Tab.3, Overall↓)**: DUSt3R 1.005(~7s), MASt3R 0.826(~9s) → VGGT 포인트맵 헤드 0.709, **depth×camera 조합 0.677(~0.2s)**.

**ScanNet-1500 2뷰 매칭 (Tab.4, AUC@5/10/20↑)**: 매칭 전용이 아닌데도 **33.9/55.2/73.4**로 SOTA인 Roma(31.8/53.4/70.9) 추월.

**백본 ablation (Tab.5, ETH3D, 파라미터 수 동일)**:

| 백본 | Acc↓ | Comp↓ | Overall↓ |
|---|---|---|---|
| Cross-Attention | 1.287 | 0.835 | 1.061 |
| Global Self-Attention만 | 1.032 | 0.621 | 0.827 |
| **Alternating-Attention** | **0.901** | **0.518** | **0.709** |

**Multi-task ablation (Tab.6, ETH3D Overall↓)**: 전 손실 0.709 vs 카메라 손실 제거 0.834, depth 제거 0.727, track 제거 0.790 — 카메라 추정 공동 학습이 포인트맵 정확도에 가장 크게 기여.

**다운스트림**: ① CoTracker2 백본을 VGGT로 교체·미세조정 시 TAP-Vid RGB-S $\delta^{vis}_{avg}$ 78.9→**84.0**, DAVIS AJ 61.8→64.7 — 동적 장면 학습 없이도 트래커가 강해진다. ② LVSM식 feed-forward NVS로 미세조정 시 GSO에서 PSNR **30.41**/LPIPS 0.033 — **입력 카메라 파라미터 없이**, LVSM 학습 데이터의 ~20%만 쓰고 LVSM(31.71, 카메라 기지)에 근접.

## 내 실습 연결

- **석사 대응점 연구(29.1→2.23px)와의 대비**: 나는 대응점 정밀도를 끌어올려 그 위의 기하 최적화가 잘 돌게 만드는 쪽이었다. VGGT는 그 층 구조 자체를 접는다 — correspondence·triangulation·BA가 전부 attention 안으로 들어가고, 기하는 손실 설계(첫 카메라 기준계, over-complete 감독)로만 주입된다. 다만 BA를 얹으면 여전히 오른다는 결과(85.3→93.5)는 "기하 최적화가 죽었다"가 아니라 **초기화가 공짜가 됐다**는 뜻으로 읽는 게 정확하다.
- **리콘랩스 COLMAP→3DGS 프로덕션 관점**: 캡처 서비스에서 COLMAP은 (1) textureless·반복 텍스처·저중첩 입력에서의 등록 실패와 (2) 처리 시간이 파이프라인 병목이었다. VGGT는 정확히 그 두 축을 친다 — Fig.3의 무중첩 2뷰·사막류 반복 텍스처에서 DUSt3R가 무너질 때도 동작하고, 수십 장을 1초 안에 처리한다. 실패 케이스 재시도 루프와 대기 시간이 곧 비용인 서비스에서 "SfM이 항상 fallback 없이 끝난다"는 것의 운영적 가치가 크다.
- **3DGS 초기화**: 저자들 스스로 포인트맵을 "3DGS 등 다운스트림과 바로 이어지는 over-parameterized 표현"으로 언급한다. VGGT의 포즈 + dense 포인트(+ confidence 맵으로 필터링)는 COLMAP sparse 포인트보다 밀도 높은 GS 초기화 후보이고, 필요하면 VGGT+BA(~1.8s)로 정밀 포즈까지 확보한 뒤 GS 학습에 넘기는 하이브리드가 현실적 배포 형태로 보인다. 로보틱스 쪽으로는 — 내가 준비 중인 camera-to-robot guidance에서 멀티뷰 한 번 촬영 → 즉시 장면 포인트맵/포즈를 얻는 perception 프런트엔드로 쓸 수 있다는 점이 매력적이다.

## 한 줄 평 / 한계

**한 줄 평**: "3D 전용 아키텍처는 필요 없다, 데이터와 감독 설계가 기하를 대신한다"를 수치로 증명한 논문 — DUSt3R가 연 포인트맵 패러다임을 쌍 단위의 한계에서 해방시켰고, over-complete 예측·추론 시 depth×camera 재조합 같은 실전적 발견이 특히 값지다.

**한계(저자 명시)**: fisheye/파노라마 미지원, 극단적 회전에서 성능 저하, 큰 비강체 변형 실패. 미분가능 BA는 학습 스텝을 ~4배 느리게 해 채택하지 않았다(향후 무감독 학습 신호로 유망하다고만 언급). 내 관점의 추가 의문: ① 200장 40GB의 global attention 메모리는 수백~수천 장짜리 실제 캡처 세션엔 여전히 벽이라 청크 분할/토큰 병합 없이는 COLMAP을 완전 대체하지 못한다. ② 주점 중심 고정·단일 카메라 모델 가정은 왜곡 큰 실촬영 입력에서 캘리브레이션 정밀도의 상한이 될 수 있다 — BA 후처리가 이걸 얼마나 흡수하는지는 논문이 분리해서 보여주지 않는다.
