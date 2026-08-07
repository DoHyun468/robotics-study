# SuperGlue (CVPR 2020)

*매칭을 "더 좋은 descriptor 학습"이 아니라 "미분 가능한 assignment 최적화"로 다시 정의한 learnable middle-end.*

Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, *SuperGlue: Learning Feature Matching with Graph Neural Networks*, CVPR 2020. ([arXiv](https://arxiv.org/abs/1911.11763), [코드](https://github.com/magicleap/SuperGluePretrainedNetwork))

## 한 줄 요약
> 두 이미지의 local feature 집합 사이 매칭을 attentional GNN이 비용을 예측하는 differentiable optimal transport(partial assignment) 문제로 풀어, 매칭·필터링·컨텍스트 집계를 하나의 end-to-end 네트워크로 통합했다. detector(front-end)와 pose estimator(back-end) 사이의 **middle-end**라는 위치 선언이 이 논문의 핵심 프레이밍이다.

## 문제 — 왜 어려웠나

SLAM/SfM의 데이터 연관(data association)은 전통적으로 i) keypoint 검출, ii) descriptor 계산, iii) NN 매칭, iv) ratio test·mutual check·GMS 같은 휴리스틱 필터, v) RANSAC의 5단 파이프라인이었다. 이 구조의 문제:

- NN 매칭은 각 keypoint를 **독립적으로** 처리해 assignment 구조(한 점은 최대 하나의 대응, occlusion 시 무대응)를 무시한다.
- PointCN, OANet 같은 학습형 outlier rejector도 NN이 이미 만든 match 집합 위에서 inlier/outlier 분류만 하므로, NN이 버린 시각 정보를 복구할 수 없다. 저자들은 Table 1에서 이들이 "NN matcher 자체보다 더 많은 correct match를 예측하지 못한다"는 것을 보인다.
- 큰 viewpoint/조명 변화, 저텍스처, 반복 패턴, occlusion에서는 descriptor 유사도만으로 대응이 결정되지 않는다 — 사람이 두 이미지를 왔다갔다 보며 문맥으로 모호성을 푸는 것처럼, **두 집합을 동시에 보는 문맥 추론**이 필요하다.

## 방법 — 아키텍처와 수식

두 이미지 $A, B$에서 keypoint 위치 $\mathbf{p}_i := (x, y, c)_i$ (검출 confidence $c$ 포함)와 descriptor $\mathbf{d}_i \in \mathbb{R}^D$가 주어진다. $A$는 $M$개, $B$는 $N$개. 목표는 soft partial assignment 행렬 $\mathbf{P} \in [0,1]^{M \times N}$ 예측이며, 제약은 (Eq. 1):

$$
\mathbf{P}\mathbf{1}_N \le \mathbf{1}_M \quad \text{and} \quad \mathbf{P}^\top \mathbf{1}_M \le \mathbf{1}_N .
$$

두 블록으로 구성: **Attentional GNN**(matching descriptor 생성) + **Optimal Matching Layer**(Sinkhorn으로 assignment 산출).

### 1. Keypoint Encoder — 위치와 외형의 결합

초기 표현은 descriptor에 위치의 MLP 임베딩을 더한다 (Eq. 2):

$$
{}^{(0)}\mathbf{x}_i = \mathbf{d}_i + \mathrm{MLP}_{\mathrm{enc}}(\mathbf{p}_i).
$$

언어 모델의 positional encoder에 해당하며, 이후 attention이 외형과 기하를 **동시에** 추론할 수 있게 한다. 부록 C: 이 MLP는 5층, 차원 $(32, 64, 128, 256, D)$, 약 100k 파라미터.

### 2. Multiplex GNN — self/cross attention 교대

두 이미지의 keypoint 전체를 노드로 하는 complete multiplex graph를 만든다. 같은 이미지 내부를 잇는 **self edge** $\mathcal{E}_{\mathrm{self}}$와 상대 이미지의 모든 keypoint를 잇는 **cross edge** $\mathcal{E}_{\mathrm{cross}}$의 두 종류. residual message passing 업데이트는 (Eq. 3):

$$
{}^{(\ell+1)}\mathbf{x}_i^A = {}^{(\ell)}\mathbf{x}_i^A + \mathrm{MLP}\!\left(\left[{}^{(\ell)}\mathbf{x}_i^A \,\|\, \mathbf{m}_{\mathcal{E}\rightarrow i}\right]\right),
$$

여기서 $[\cdot\|\cdot]$은 concatenation. $\ell$이 홀수면 $\mathcal{E} = \mathcal{E}_{\mathrm{self}}$, 짝수면 $\mathcal{E} = \mathcal{E}_{\mathrm{cross}}$로 교대한다. 메시지는 attention 가중 평균 (Eq. 4):

$$
\mathbf{m}_{\mathcal{E}\rightarrow i} = \sum_{j:(i,j)\in\mathcal{E}} \alpha_{ij}\mathbf{v}_j, \qquad \alpha_{ij} = \mathrm{Softmax}_j\!\left(\mathbf{q}_i^\top \mathbf{k}_j\right).
$$

query는 자기 이미지 $Q$의 keypoint $i$에서, key/value는 source 이미지 $S \in \{A, B\}$의 keypoint들에서 선형 사영으로 계산 (Eq. 5):

$$
\mathbf{q}_i = \mathbf{W}_1\, {}^{(\ell)}\mathbf{x}_i^Q + \mathbf{b}_1, \qquad
\begin{bmatrix}\mathbf{k}_j \\ \mathbf{v}_j\end{bmatrix} =
\begin{bmatrix}\mathbf{W}_2 \\ \mathbf{W}_3\end{bmatrix} {}^{(\ell)}\mathbf{x}_j^S +
\begin{bmatrix}\mathbf{b}_2 \\ \mathbf{b}_3\end{bmatrix}.
$$

self-attention은 같은 이미지의 salient한 점들에, cross-attention은 상대 이미지의 외형이 비슷한 후보들에 attend한다(Figure 4). 부록 D의 정량 분석: attention span은 층이 깊어질수록 10배 이상 줄어들며, 초반엔 넓게 훑고 후반엔 진짜 match 근방으로 좁혀 들어간다 — "back-and-forth" 시선의 학습된 대응물. 최종 matching descriptor는 (Eq. 6):

$$
\mathbf{f}_i^A = \mathbf{W}\cdot {}^{(L)}\mathbf{x}_i^A + \mathbf{b}, \quad \forall i \in \mathcal{A}.
$$

### 3. Optimal Matching Layer — score, dustbin, Sinkhorn

pairwise score는 matching descriptor의 내적 (Eq. 7):

$$
\mathbf{S}_{i,j} = \langle \mathbf{f}_i^A, \mathbf{f}_j^B \rangle, \ \forall (i,j) \in \mathcal{A}\times\mathcal{B}.
$$

matching descriptor는 (학습된 visual descriptor와 달리) 정규화하지 않는다 — 크기 자체가 예측 confidence로 학습되게 놔둔다. occlusion/미검출 keypoint를 흡수하기 위해 score 행렬에 행·열을 하나씩 붙인 **dustbin** 확장 $\bar{\mathbf{S}}$를 만들고, 새 엔트리는 학습 가능한 스칼라 하나로 채운다 (Eq. 8):

$$
\bar{\mathbf{S}}_{i,N+1} = \bar{\mathbf{S}}_{M+1,j} = \bar{\mathbf{S}}_{M+1,N+1} = z \in \mathbb{R}.
$$

각 dustbin은 상대 집합 크기만큼 match를 받을 수 있으므로, 기대 매칭 수 벡터 $\mathbf{a} = [\mathbf{1}_M^\top \ N]^\top$, $\mathbf{b} = [\mathbf{1}_N^\top \ M]^\top$에 대해 확장 assignment $\bar{\mathbf{P}}$의 제약은 (Eq. 9):

$$
\bar{\mathbf{P}}\mathbf{1}_{N+1} = \mathbf{a} \quad \text{and} \quad \bar{\mathbf{P}}^\top\mathbf{1}_{M+1} = \mathbf{b}.
$$

이 문제는 이산 분포 $\mathbf{a}, \mathbf{b}$ 사이의 **optimal transport**이고, entropy-regularized 형태라 **Sinkhorn 알고리즘**으로 GPU에서 효율적으로 푼다. Sinkhorn은 Hungarian 알고리즘의 미분 가능 버전으로, $\exp(\bar{\mathbf{S}})$를 행과 열 방향으로 번갈아 정규화(row/column Softmax와 유사)하는 반복이다. $T$회 반복 후 dustbin을 떼어 $\mathbf{P} = \bar{\mathbf{P}}_{1:M,1:N}$을 회수한다. (공식 구현은 수치 안정성을 위해 이 반복을 log-domain에서 수행한다.)

### 4. 손실 — assignment에 대한 NLL

GNN과 matching layer가 모두 미분 가능하므로 match에서 descriptor까지 역전파된다. GT 상대 변환(pose+depth 또는 homography)의 재투영으로 GT match $\mathcal{M} = \{(i,j)\} \subset \mathcal{A}\times\mathcal{B}$와 unmatched 집합 $\mathcal{I} \subseteq \mathcal{A}$, $\mathcal{J} \subseteq \mathcal{B}$를 라벨링하고, 확장 assignment의 negative log-likelihood를 최소화한다 (Eq. 10):

$$
\mathrm{Loss} = -\sum_{(i,j)\in\mathcal{M}} \log\bar{\mathbf{P}}_{i,j}
\;-\sum_{i\in\mathcal{I}} \log\bar{\mathbf{P}}_{i,N+1}
\;-\sum_{j\in\mathcal{J}} \log\bar{\mathbf{P}}_{M+1,j}.
$$

match 항은 recall을, dustbin 항은 precision을 동시에 밀어올린다. mutual check 같은 휴리스틱이 하던 상호성(reciprocity)이 optimal transport 제약으로 **soft하게 학습 과정 안에** 들어온다는 점, 그리고 아키텍처가 keypoint 순서뿐 아니라 **이미지 순서에도 equivariant**하다는 점이 설계상 미덕이다.

### 하이퍼파라미터 (원문 수치)

- 모든 중간 표현·descriptor 차원 $D = 256$ (SuperPoint와 동일)
- self/cross 교대 attention $L = 9$ 층, multi-head **4 heads**, Sinkhorn $T = 100$ 반복
- 총 18개 층(부록 C 기준), **12M 파라미터**; message update MLP는 2층 $(2D, D)$, 층당 0.66M
- 추론: GTX 1080에서 실내 이미지 쌍 평균 **69 ms (15 FPS)**. 부록 C: keypoint 512개일 때 69 ms, 1024개 87 ms, 2048개 270 ms — GNN과 matching layer의 비용이 비슷한 수준
- test 시 confidence threshold 0.2로 match 선별

## 결과 — 원문 테이블 수치

**Homography estimation** (Oxford-Paris 1M distractor 이미지 + 합성 homography, SuperPoint feature, Table 1): AUC (RANSAC / DLT)

| Matcher | RANSAC | DLT | P | R |
|---|---|---|---|---|
| NN | 39.47 | 0.00 | 21.7 | 65.4 |
| NN + PointCN | 43.02 | 45.40 | 76.2 | 64.2 |
| NN + OANet | 44.55 | 52.29 | 82.8 | 64.7 |
| **SuperGlue** | **53.67** | **65.85** | **90.7** | **98.3** |

recall 98.3%로 "가능한 match를 거의 전부" 회수하며, correspondence 품질이 좋아 강건 추정기 없는 DLT가 RANSAC을 능가한다 — outlier가 애초에 거의 없다는 뜻. (HPatches는 별도 정량 테이블 없이 Figure 13의 정성 비교에만 등장한다.)

**ScanNet indoor wide-baseline pose estimation** (1500 test pairs, pose error AUC, Table 2):

| Features | Matcher | AUC@5° | @10° | @20° | P | MS |
|---|---|---|---|---|---|---|
| SIFT | NN + ratio test | 5.83 | 13.06 | 22.47 | 40.3 | 1.0 |
| SIFT | NN + OANet | 6.00 | 14.33 | 25.90 | 38.6 | 4.2 |
| SIFT | **SuperGlue** | **6.71** | **15.70** | **28.67** | 74.2 | 9.8 |
| SuperPoint | NN + mutual | 9.43 | 21.53 | 36.40 | 50.4 | 18.8 |
| SuperPoint | NN + PointCN | 11.40 | 25.47 | 41.41 | 71.8 | 25.5 |
| SuperPoint | NN + OANet | 11.76 | 26.90 | 43.85 | 74.0 | 25.7 |
| SuperPoint | **SuperGlue** | **16.16** | **33.81** | **51.84** | **84.4** | **31.5** |

SuperPoint 기준 최강 학습형 baseline OANet 대비 AUC@20° 43.85→51.84 (+8.0pt), AUC@5°는 11.76→16.16으로 1.37배. SIFT에 붙였을 때 ratio test 대비 correct match 수는 최대 10배.

**PhotoTourism outdoor pose estimation** (Table 3):

| Features | Matcher | AUC@5° | @10° | @20° | P | MS |
|---|---|---|---|---|---|---|
| SIFT | NN + ratio test | 15.19 | 24.72 | 35.30 | 43.4 | 1.7 |
| SIFT | **SuperGlue** | **23.68** | **36.44** | **49.44** | 74.1 | 7.2 |
| SuperPoint | NN + OANet | 21.03 | 34.08 | 46.88 | 52.4 | 8.4 |
| SuperPoint | **SuperGlue** | **34.18** | **50.32** | **64.16** | **84.9** | **11.1** |

precision 84.9% — "glue"라는 이름값을 하는 수치. 부록 B의 Aachen Day-Night visual localization에서는 keypoint 4k개만으로 (0.5m/2°, 1m/5°, 5m/10°) 기준 45.9 / **70.4** / 88.8%를 기록, 15–20k 특징을 쓰는 R2D2·D2-Net과 대등하거나 앞선다(Table 7).

**Ablation** (ScanNet + SuperPoint, Table 4, AUC@20° / P / MS):

| Variant | AUC@20° | P | MS |
|---|---|---|---|
| NN + mutual (baseline) | 36.40 | 50.4 | 18.8 |
| No Graph Neural Net (Sinkhorn만) | 38.56 | 66.0 | 17.2 |
| No cross-attention | 42.57 | 74.0 | 25.3 |
| No positional encoding | 47.12 | 75.8 | 26.6 |
| Smaller (3 layers) | 46.93 | 79.9 | 30.0 |
| **Full (9 layers)** | **51.84** | **84.4** | **31.5** |

optimal matching layer 단독 이득(+2.2pt)보다 GNN의 기여가 압도적이고, 그중에서도 cross-attention과 positional encoding이 결정적이다. SuperPoint descriptor 네트워크까지 역전파하는 end-to-end 학습 시 AUC@20°가 51.84→53.38로 추가 상승 — matching을 넘어선 end-to-end 학습 가능성의 신호.

## 내 실습 연결

석사 논문에서 대형 선체블록의 wide-baseline 스테레오 대응점 탐지를 다루며 keypoint 매칭 단계에 SuperGlue·DKM·LoFTR 앙상블을 실제로 사용했고, 신뢰도·epipolar·재투영의 3단 필터링과 결합해 평균 대응 오차를 29.1px→2.23px로 낮췄다. 그 경험에서 본 SuperGlue의 실전 프로파일은 이 논문의 설계와 정확히 맞물린다. 선체 표면은 용접선·격자 보강재 같은 **반복 구조**와 넓은 **저텍스처 도장면**이 공존하는데, descriptor만 보는 NN 매칭이 반복 패턴에서 체계적으로 미끄러지는 반면 SuperGlue는 self-attention이 실어주는 이미지 내 상대 기하(positional encoding + 문맥) 덕분에 "몇 번째 격자인지"를 구분하는 데 강했다 — Figure 10에서 저자들이 창문 반복 패턴의 건물 파사드를 정확히 매칭하는 것과 같은 메커니즘이다. 반면 약점도 뚜렷했다. SuperPoint가 keypoint를 아예 못 찍는 균질한 도장면에서는 middle-end인 SuperGlue가 개입할 여지 자체가 없어(detector 의존성), dense/semi-dense 방식인 LoFTR·DKM이 그 빈자리를 메웠다. 앙상블에서 SuperGlue의 역할은 "구조가 있는 영역에서 정밀도 높은 sparse anchor를 공급"하는 쪽이었고, matching descriptor의 비정규화 크기에서 나오는 confidence(threshold 0.2 방식)는 3단 필터의 첫 관문으로 그대로 쓰기 좋았다. wide-baseline에서 mutual check보다 optimal transport 제약이 훨씬 부드럽게 일대일성을 유지해 주는 것도 체감 포인트였다.

## 한 줄 평 / 한계

매칭 문제의 무게중심을 descriptor에서 assignment 구조로 옮긴, sparse matching 계보의 분수령. 강점은 (i) 문맥 집계·매칭·필터링을 하나의 미분 가능 파이프라인으로 묶은 문제 정식화, (ii) dustbin+Sinkhorn으로 occlusion을 우아하게 다루는 partial assignment, (iii) SIFT든 SuperPoint든 front-end를 가리지 않는 drop-in 호환성. 한계는 명확하다. complete graph attention의 비용이 keypoint 수에 대해 나쁘게 스케일링되어(512개 69 ms → 2048개 270 ms) 저자들 스스로 >500 keypoint에선 기존 graph matching이 불가능하다고 언급할 만큼 계산량이 본질적 제약이고, detector가 keypoint를 못 찾는 저텍스처 영역은 구조적으로 손댈 수 없다(Figure 14의 "Too Difficult" 실패 사례도 반복성 keypoint 부재가 원인). 바로 이 두 지점 — attention 기반 문맥 매칭이라는 유산과 detector 의존이라는 공백 — 이 이후 LoFTR을 위시한 detector-free dense matching과 효율화 후속(LightGlue)으로 이어졌다. "learnable middle-end"라는 프레임과 Sinkhorn 기반 assignment layer는 이후 매칭 연구의 표준 어휘가 됐다.
