# DKM: Dense Kernelized Feature Matching (CVPR 2023)

*"매칭을 검출 문제가 아니라 회귀 문제로" — Gaussian Process로 좌표 임베딩을 회귀하는 dense matcher가 sparse의 아성(SuperGlue·LoFTR 계열)을 pose 추정에서 처음으로 넘어선 논문*

Johan Edstedt, Ioannis Athanasiadis, Mårten Wadenbäck, Michael Felsberg, *DKM: Dense Kernelized Feature Matching for Geometry Estimation*, CVPR 2023. Linköping University.
([arXiv:2202.00667](https://arxiv.org/abs/2202.00667) · [코드](https://github.com/Parskatt/dkm))

## 한 줄 요약

> 픽셀 전부를 매칭하는 dense warp 추정을 **GP 회귀로 정식화한 kernelized global matcher** + **stacked feature map·5×5 depthwise 커널 warp refiner** + **depth-consistency로 학습한 certainty와 KDE 균형 샘플링** 세 축으로 재설계해, MegaDepth-1500 AUC@5°에서 최고 sparse 대비 **+4.9**, 최고 dense 대비 **+8.9**를 얻으며 "dense는 geometry 추정에서 sparse보다 못하다"는 통념을 뒤집었다.

## 문제

두 뷰 geometry 추정(relative pose, homography)의 첫 단계는 feature matching이다. 접근은 세 갈래다.

- **Sparse** (SuperPoint+SuperGlue): 키포인트 검출→기술→매칭. 어려운 장면에서 반복 검출 가능한 키포인트의 정확한 localization 자체가 난제.
- **Semi-dense / detector-free** (LoFTR, Patch2Pix, ASpanFormer): 검출 없이 coarse grid에서 전역 매칭 후 sparse refinement. 저텍스처에 강하지만 coarse scale에서 매치가 생성되어 grid artifact·반복성 문제가 남는다.
- **Dense** (GLU-Net, PDC-Net+): 모든 픽셀의 warp + certainty를 회귀. 부분픽셀 정밀도와 affine 매치(더 작은 minimal solver)가 공짜로 나오지만, 종래에는 geometry 추정 성능이 sparse에 밀렸다.

DKM은 dense가 밀린 이유를 (1) global matcher의 구조, (2) refiner의 표현력, (3) certainty의 품질과 매치 샘플링 세 곳에서 찾아 각각 고친다.

## 방법

파이프라인 5단계: **I** 인코더 → **II** global matcher $G_\theta$ → **III** warp refiner $R_\theta$ → **IV** certainty 기반 매치 샘플링 → **V** RANSAC + minimal solver.

### I. 인코더

ImageNet-1K 사전학습 ResNet50, 두 이미지 가중치 공유. multiscale 피처 $\{\varphi_l\}_{l=1}^{L}$에서 coarse는 stride $\{32,16\}$, fine은 stride $\{8,4,2,1\}$.

### II. Kernelized Global Matcher — 매칭을 GP 회귀로

핵심 발상: 이미지 $\mathcal{A}$의 각 픽셀에 대해 "$\mathcal{B}$의 어느 좌표에 대응하는가"를 매핑 $\varphi \to \chi$ ($\chi$는 $\mathcal{B}$의 좌표 임베딩)의 **회귀 문제**로 본다. 회귀기는 GP: 출력 $\chi \in \mathbb{R}^{H\cdot W \times C}$를 jointly Gaussian 확률변수로 두고, 커널은 지수 코사인 유사도

$$k(\varphi,\varphi') = \exp(-\tau)\exp\!\Big(\tau \frac{\langle\varphi,\varphi'\rangle}{\sqrt{\langle\varphi,\varphi\rangle\langle\varphi',\varphi'\rangle+\varepsilon}}\Big)$$

$\tau=5$ 고정(학습해도 이득 없음, $\tau\in[3,10]$에서 강건), $\varepsilon=10^{-6}$. squared exponential도 초기 실험에서 비슷했다고 한다. $(\varphi^{\mathcal{B}}_{\text{coarse}}, \chi^{\mathcal{B}}_{\text{coarse}})$가 i.i.d. 노이즈로 관측된다는 표준 가정하에 posterior는 폐형식

$$\mu(\varphi^{\mathcal{A}}|\varphi^{\mathcal{B}}) = K^{AB}(K^{BB}+\sigma_n^2 I)^{-1}\chi^{\mathcal{B}}, \qquad \Sigma = K^{AA} - K^{AB}(K^{BB}+\sigma_n^2 I)^{-1}K^{BA}$$

($\sigma_n=0.1$). 즉 종래의 4D correlation volume 회귀를, 커널 조건화된 확률적 회귀로 대체한 것.

**좌표 임베딩**: raw 2D 좌표를 회귀하면 GP posterior가 unimodal이라 다봉성(반복 구조)에서 붕괴한다. 그래서 좌표를 코사인 임베딩

$$B_{\mathcal{F}}(x; W, b) = \cos(Wx+b), \quad W_{ij}\sim\mathcal{N}(0,\ell^2),\ b_i\sim\mathcal{U}_{[0,2\pi)}$$

으로 올려 multimodality를 보존한다(random Fourier features 계열).

**Embedding decoder $D_\theta$**: posterior 평균을 grid로 reshape한 $\mu_{\text{grid}} \in \mathbb{R}^{H\times W\times C}$와 $\varphi^{\mathcal{A}}_{\text{coarse}}$를 받아, canonical grid $[-1,1]^2$ 좌표(=coarse warp)와 픽셀별 매치 유효성 logit(=coarse certainty)을 디코딩한다. global matcher는 stride 32와 16 두 층에 두고, stride 16 디코더는 stride 32 디코더의 context feature를 받는다.

### III. Warp Refinement — stacked feature map + depthwise 5×5

coarse warp/certainty를 bilinear upsample하며 각 fine stride $l\in\{8,4,2,1\}$에서 residual을 재귀적으로 예측:

$$(\hat W^{\mathcal{A}\to\mathcal{B}}_l, \hat p^{\mathcal{A}\to\mathcal{B}}_l) = R_{\theta,l}(\varphi^{\mathcal{A}}_l, \varphi^{\mathcal{B}}_l, \hat W_{l+1}, \hat p_{l+1})$$

선행 dense 연구(PDC-Net 계열) 대비 두 가지 개선.

- **입력 표현**: 종래는 $\mathcal{A}$ 피처 + $\mathcal{A}$ 기준 local correlation만 썼지만, DKM은 warp로 정렬한 **$\mathcal{B}$ 피처맵 전 채널을 그대로 concat**(stacked feature maps)하고 local correlation도 $\mathcal{B}$ 쪽에서 계산.
- **블록 구조**: DenseNet식 3×3 대신 **5×5 depthwise separable conv + 1×1 conv** (ReLU, BatchNorm) 블록을 **scale당 8개** 스택. warp 자체는 displacement로 변환해 선형 임베딩 후 함께 입력.

### IV. Certainty 학습과 Balanced Sampling

**Certainty = depth-consistency 분류**: MegaDepth의 SfM depth를 이용해, $\mathcal{A}\to\mathcal{B}$ warp 후 상대 depth 일관성

$$p^{\mathcal{A}\to\mathcal{B}} = \Big|\frac{z^{\mathcal{A}\to\mathcal{B}} - z^{\mathcal{B}}}{z^{\mathcal{B}}}\Big| < \alpha, \quad \alpha=0.05$$

를 만족하는 픽셀을 positive로 하는 이진 분류로 certainty를 학습한다. PDC-Net+의 mixture 기반 확률 모델이 unmatchable 영역에서도 과신하는 것과 대비되는 지점.

**Balanced sampling**: certainty 0.05로 threshold 후 certainty를 가중치로 대량 샘플링 → 4차원 매치 공간에서 **KDE**를 계산 → 각 매치를 KDE의 역수로 재가중해 장면 전체에 고르게 분포한 매치 집합을 만든다. RANSAC/5-point가 잘 분포된 매치를 선호한다는 점을 겨냥. 추론 시 최대 5000 매치 샘플, 양방향 warp를 단순 concat한 bidirectional 매칭 사용.

### 손실·학습

$$\mathcal{L} = \sum_{l=1}^{L} \mathcal{L}_{\text{warp}}(\hat W_l) + \lambda\,\mathcal{L}_{\text{conf}}(\hat p_l), \quad \lambda=0.01$$

warp 손실은 GT consistent-depth mask $p_l$로 가중한 $\ell_2$, certainty 손실은 비가중 BCE. fine stride에서는 coarse warp가 GT에서 임계 이상 벗어나면 $p$를 0으로 두고, scale 간 gradient는 detach. 학습: batch 32, AdamW(weight decay $10^{-2}$), lr $4\cdot10^{-4}$(decoder/refiner)·$2\cdot10^{-5}$(backbone), 250k step(166,666·225,000 step에서 ×0.2 감쇠), A100 4장으로 약 5일. Outdoor는 MegaDepth 540×720, indoor는 ScanNet 480×640 추가 학습. 평가는 5회 벤치 평균.

## 결과 (원문 수치)

**MegaDepth-1500 relative pose, AUC@5°/10°/20°** (Tab. 2):

| Method | @5° | @10° | @20° |
|---|---|---|---|
| SuperGlue | 42.2 | 61.2 | 76.0 |
| LoFTR | 52.8 | 69.2 | 81.2 |
| ASpanFormer (최고 sparse) | 55.3 | 71.5 | 83.1 |
| PDC-Net+ (최고 dense) | 51.5 | 67.2 | 78.5 |
| **DKM** | **60.4** | **74.9** | **85.1** |

최고 sparse 대비 **+4.9 AUC@5°**, 최고 dense 대비 **+8.9 AUC@5°**.

- **ScanNet-1500** (indoor): DKM **29.4 / 50.7 / 68.3** — ASpanFormer(25.6) 대비 +4.0, PDC-Net+(20.3) 대비 +9.3 AUC@5°.
- **HPatches homography**: AUC@3px **71.3** (PDC-Net+ 67.7 대비 +3.6), @5px 80.6, @10px 88.5.
- **MegaDepth-8-Scenes** (저자 신설, 8개 장면 1600쌍): DKM **60.5** AUC@5° vs ASpanFormer 57.2, PDC-Net+ 51.8.
- **St. Paul's Cathedral** (ECO-TR 프로토콜): mAA@5° **53.3** vs ECO-TR 45.3, COTR 44.3 (+8.0).

**Ablation** (MegaDepth, AUC@5° 기준):

- Global matcher: correlation-volume 회귀 baseline 57.0 → 코사인 임베딩 GP **58.1** (+1.1). 선형 좌표 임베딩은 57.9로 코사인보다 낮음 — 다봉성 보존의 효과.
- Warp refiner: depthwise refiner가 baseline 대비 **+4.8**, stacked FM 입력 표현이 **+1.5** — 세 기여 중 refiner의 몫이 가장 크다.
- Sampling: certainty 없이 42.9 → certainty 샘플링 56.1 → balanced **58.1** (+2.0). certainty가 없으면 dense는 무너진다는 것을 수치로 보여준다.
- 해상도 384×512→540×720 +1.3, bidirectional +1.0.

## 내 실습 연결

석사에서 선체블록 wide-baseline 스테레오 대응에 SuperGlue·**DKM**·LoFTR 앙상블을 실사용했다(대응 오차 29.1→2.23px). 그 경험에서 본 DKM의 실용 가치는 정확도 수치보다 **출력의 형태**에 있다.

- **dense + certainty라는 출력 조합**이 앙상블·필터링 파이프라인과 궁합이 좋다. sparse 매처는 "여기서 매치가 났다/안 났다"만 주지만, DKM은 모든 픽셀에 warp와 신뢰도를 주므로 임계값을 파이프라인 쪽에서 정할 수 있다 — SuperGlue·LoFTR 매치와 교차 검증하거나, 신뢰도 기반 필터로 outlier를 걸러내는 단계에 dense certainty map이 그대로 재료가 된다. 선체블록처럼 반복 구조·저텍스처가 섞인 산업 장면에서 단일 매처의 실패 모드를 다른 매처로 메우는 앙상블에는, "어디를 믿으면 안 되는지"를 말해주는 매처가 하나 있어야 한다.
- 원문 Fig. 6의 지적 — PDC-Net+는 unmatchable 영역에서도 과신한다 — 는 실무에서 그대로 체감한 문제다. depth-consistency 분류로 학습한 DKM certainty는 threshold(0.05) 하나로 안정적으로 걸러졌다.
- balanced sampling(KDE 역수 가중)은 앙상블 이후 RANSAC 단계에도 이식할 수 있는 아이디어다. 매치가 특정 영역에 몰리면 E-matrix 추정이 ill-conditioned가 되는 것은 매처와 무관한 공통 문제.

## 한 줄 평 / 한계

> "dense는 느리고 부정확하다"던 시절을 끝낸 논문 — GP 정식화 자체보다, certainty를 depth-consistency 분류로 학습하고 샘플링까지 추정기 관점에서 설계한 시스템 감각이 승부처였고, 이 설계는 그대로 후속작 RoMa의 뼈대가 된다.

한계는 원문이 스스로 명시한다. (1) global matcher는 다봉성을 다루지만 **warp refinement는 unimodal**이라 depth 경계(불연속)에서 흔들린다(Fig. 7). (2) 하늘과 맞닿은 작은 물체에 과도하게 낮은 certainty — consistent depth "분류" 학습의 부작용으로, PDC-Net식 모델 불확실성 예측과의 trade-off. (3) 텍스처가 극단적으로 없으면 warp가 완전히 실패하지만, 이때 certainty도 매우 낮게 나와 보정은 잘 되어 있다(Fig. 18). (4) ablation을 뜯어보면 GP global matcher의 기여(+1.1)는 refiner(+4.8)나 certainty 샘플링(certainty 유무로 +13 이상)보다 작다 — 논문 제목의 "kernelized"가 성능의 주역은 아니라는 점은 읽는 쪽에서 균형 있게 봐야 한다. 학습 비용(A100 4장 5일)도 LoFTR(1080ti 64장 1일) 대비 가벼운 편은 아니다.
