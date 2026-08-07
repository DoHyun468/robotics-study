# CAPS (ECCV 2020)

*dense correspondence GT 없이 — 카메라 포즈가 주는 epipolar line만으로 descriptor를 학습하다*

Qianqian Wang, Xiaowei Zhou, Bharath Hariharan, Noah Snavely, *Learning Feature Descriptors using Camera Pose Supervision*, ECCV 2020 (arXiv 2004.13324). Cornell + Cornell Tech + Zhejiang University. ([project/code](https://qianqianwang68.github.io/CAPS/)) 내 석사 연구에서 COTR과 함께 기존 방법 베이스라인으로 직접 돌려본 모델.

## 한 줄 요약

> descriptor 학습의 병목인 pixel-level GT correspondence를 버리고, **상대 카메라 포즈만으로**(weakly-supervised) 학습한다 — 포즈에서 나오는 fundamental matrix F로 "예측 대응점이 epipolar line 위에 있어야 한다"는 제약을 손실로 바꾸고, correspondence를 2D softmax 분포의 **기대값**으로 정의해 전체를 미분 가능하게 만든 뒤, coarse-to-fine 매칭으로 계산량을 줄였다. weak supervision만으로 fully-supervised descriptor들과 대등하거나 그 이상.

## 문제

학습형 descriptor는 표준 벤치마크에서 hand-crafted를 넘었지만, 실세계 unseen 데이터로 가면 일반화가 자주 무너진다. 논문이 지목하는 원인은 **학습 데이터의 양과 다양성 부족**이고, 그 근원은 supervision의 형태다.

- **dense GT correspondence**: 실사 이미지에서 수집이 극히 어려워 이런 데이터셋 자체가 몇 개 없다.
- **SfM pseudo-GT**: 재구성된 특징점 매칭을 GT로 쓰는 방식 — sparse하고, SfM 파이프라인이 쓴 keypoint에 편향된다.
- **homography 합성쌍**: 실사의 기하·광학 변화 범위를 담지 못한다.

반면 **카메라 포즈**는 IMU/GPS 같은 비-비전 센서로도, SfM으로도 신뢰성 있게 대량으로 얻을 수 있다. 문제는 기존 metric learning 틀(triplet/contrastive loss)이 포즈로는 정의되지 않는다는 것 — "이 두 픽셀이 매칭이다/아니다"를 모르기 때문이다. CAPS는 supervision 요구를 포즈로 낮추기 위해 손실과 아키텍처를 함께 새로 설계한다.

## 방법

### Epipolar loss + cycle consistency loss — 손실 구성 전체

이미지 쌍 I₁, I₂와 상대 포즈·intrinsics에서 fundamental matrix F를 계산한다. epipolar constraint x₂ᵀF x₁ = 0에서, F x₁은 x₁에 대응하는 I₂ 위의 epipolar line이다. 예측 대응 함수 h₁→₂를 두고, **예측 대응점과 GT epipolar line 사이 거리**를 손실로 삼는다 (Eq. 1, 원문 그대로):

$$\mathcal{L}_{ep}(\mathbf{x}_1) = \mathrm{dist}\big(h_{1\to2}(\mathbf{x}_1),\, \mathbf{F}\mathbf{x}_1\big)$$

dist(·,·)는 점–직선 거리. 그런데 epipolar loss만으로는 예측이 "선 위 어딘가"에만 놓이면 되므로(진짜 대응 위치는 선 위 어느 점인지 모름), **forward-backward 매핑이 제자리로 돌아와야 한다**는 cycle consistency를 추가한다 (Eq. 2):

$$\mathcal{L}_{cy}(\mathbf{x}_1) = \|h_{2\to1}(h_{1\to2}(\mathbf{x}_1)) - \mathbf{x}_1\|_2$$

이 항이 epipolar 제약만 만족하는 가짜 해를 억제한다. 쌍당 n개의 query point에 대해 합산한 기본 목적식이 Eq. 3:

$$\mathcal{L}(\mathbf{I}_1,\mathbf{I}_2) = \sum_{i=1}^{n}\big[\mathcal{L}_{ep}(\mathbf{x}_1^i) + \lambda\,\mathcal{L}_{cy}(\mathbf{x}_1^i)\big]$$

### Differentiable matching layer — expectation as correspondence

손실이 예측 대응점의 **픽셀 좌표**에 걸려 있으므로, 좌표가 descriptor에 대해 미분 가능해야 한다. nearest neighbor 매칭은 미분 불가라서, 매칭을 분포의 기대값으로 다시 정의한다. 공유 가중치 CNN이 dense feature map M₁, M₂를 뽑고, query x₁의 descriptor M₁(x₁)을 M₂ 전체와 correlation한 뒤 2D softmax로 확률 분포를 만든다 (Eq. 4):

$$p(\mathbf{x}\,|\,\mathbf{x}_1,\mathbf{M}_1,\mathbf{M}_2) = \frac{\exp\!\big(\mathbf{M}_1(\mathbf{x}_1)^{\mathsf T}\mathbf{M}_2(\mathbf{x})\big)}{\sum_{\mathbf{y}\in \mathbf{I}_2}\exp\!\big(\mathbf{M}_1(\mathbf{x}_1)^{\mathsf T}\mathbf{M}_2(\mathbf{y})\big)}$$

단일 대응점은 이 분포의 기대값 — soft-argmax 형태다 (Eq. 5):

$$\hat{\mathbf{x}}_2 = h_{1\to2}(\mathbf{x}_1) = \sum_{\mathbf{x}\in \mathbf{I}_2} \mathbf{x}\cdot p(\mathbf{x}\,|\,\mathbf{x}_1,\mathbf{M}_1,\mathbf{M}_2)$$

대응 위치가 descriptor correlation에서 계산되므로, "위치를 맞히라"는 압력이 곧 descriptor 학습 신호가 된다 — end-to-end로 전체가 미분 가능.

**Uncertainty reweighting**: 분포 p의 공분산 trace를 총분산 σ²(x₁)로 정의하면 예측 신뢰도의 해석 가능한 척도가 된다(분산이 크면 multimodal/diffuse). GT 대응이 없으니 query가 occlusion·truncation으로 상대 이미지에 아예 없을 수도 있는데, 그런 점의 손실은 잘못된 신호다. 그래서 점별로 1/σ로 재가중한 최종 손실이 Eq. 6:

$$\mathcal{L}(\mathbf{I}_1,\mathbf{I}_2) = \sum_{i=1}^{n} \frac{1}{\sigma(\mathbf{x}_1^i)}\big[\mathcal{L}_{ep}(\mathbf{x}_1^i) + \lambda\,\mathcal{L}_{cy}(\mathbf{x}_1^i)\big]$$

가중치는 합이 1이 되게 정규화. 논문은 이 전략이 **빠른 수렴에 critical**했다고 명시한다(semantic correspondence 계열과 달리 별도 uncertainty 예측 네트워크 없이 learned descriptor에서 직접 유도되는 점이 차별점).

### Coarse-to-fine 아키텍처

전체 이미지에 대한 full correlation은 비싸다. flat feature map 하나 대신 **coarse level(원본의 1/16)과 fine level(1/4)** 두 해상도의 feature map을 만든다 — ImageNet pretrained ResNet-50을 layer3 뒤에서 자른 backbone에 conv를 더해 coarse feature를, up-sampling + skip-connection으로 fine feature를 얻고, 둘 다 128차원이다.

매칭은: coarse 분포 pᶜ를 **전체 위치**에 대해 계산 → 그 최빈 위치를 중심으로 fine level에 **local window W**(fine feature map 크기의 1/8)를 잘라 그 안에서만 fine 분포 pᶠ를 계산. 학습 시 두 레벨의 대응 모두에 Eq. 6 손실을 걸어 coarse/fine descriptor를 동시에 학습한다. 계산 절감뿐 아니라 매칭 정확도 자체도 올리며(fine level이 local window 안에서만 기대값을 계산해 multimodal 분포 문제가 줄어든다), 테스트 시엔 keypoint 위치에서 두 레벨 feature를 interpolation해 **연접(concatenate)한 hierarchical descriptor**를 쓰고 표준 Euclidean distance로 매칭한다 — 기존 매칭 파이프라인에 그대로 끼워 넣을 수 있다.

### 학습 설정 (원문 수치)

- **데이터**: MegaDepth — 100만+ 인터넷 사진을 COLMAP으로 재구성한 196개 scene 중 130개로 학습(나머지 val/test). 포즈와 intrinsics만 사용, 수백만 학습쌍.
- **하이퍼파라미터**: Adam, base lr 10⁻⁴; λ = 0.1; 쌍당 query point n = 500 (90%는 SIFT keypoint, 10%는 random point).

## 결과

**HPatches sparse matching (MMA)**: 116 시퀀스(illumination 57, viewpoint 59), D2-Net 프로토콜. SuperPoint keypoint와 결합 시 **2px 이후 전 threshold에서 최고 성능** (feature/match 수: SuperPoint+CAPS 1.7K/0.9K, SIFT+CAPS 4.4K/1.5K). 같은 detector 대결에서도 "SIFT+CAPS > SIFT+ContextDesc", "SuperPoint+CAPS > SuperPoint" 확인.

**HPatches dense matching (PCK)**: Dense SIFT, SuperPoint, D2-Net, R2D2 대비 전체 최고. 4px 이하 소threshold에서만 R2D2에 밀리는데, 비교된 R2D2는 full-resolution descriptor map, CAPS는 4× downsample이라는 해상도 차이 때문.

**Homography estimation, HPatches (정확도 %, ε = 1/3/5px, Table 1)**:

| Methods | ε=1 | ε=3 | ε=5 |
|---|---|---|---|
| SIFT | 40.5 | 68.1 | 77.6 |
| LF-Net | 34.8 | 62.9 | 73.8 |
| SuperPoint | 37.4 | 73.1 | 82.8 |
| D2-Net | 16.7 | 61.0 | 75.9 |
| ContextDesc | 41.0 | 73.1 | 82.2 |
| R2D2 | 40.0 | **75.0** | 84.7 |
| CAPS w/ SIFT kp. | 34.6 | 72.2 | 81.7 |
| CAPS w/ SuperPoint kp. | **44.8** | 74.5 | **85.7** |

**Relative pose estimation (rotation/translation 정확도 %, Table 2)**: MegaDepth(10° threshold)는 상대 회전각 기준 easy [0°,15°] / moderate [15°,30°] / hard [30°,60°], ScanNet(5°)은 frame 간격 10/30/60, 각 1,000쌍. essential matrix + RANSAC 후 분해.

| Methods | ScanNet d=60 | MD easy | MD moderate | MD hard |
|---|---|---|---|---|
| SIFT w/ ratio test | 44.3 / 15.9 | 63.9 / 25.6 | 36.5 / 17.0 | 20.8 / 13.2 |
| SuperPoint | 53.4 / 22.1 | 67.2 / 27.1 | 38.7 / 18.8 | 24.5 / 14.1 |
| ContextDesc | 51.4 / 18.5 | 68.9 / 27.1 | 43.1 / 21.5 | 27.5 / 14.1 |
| R2D2 | **62.9 / 28.8** | 69.4 / 30.3 | 48.3 / 23.9 | 32.6 / 17.4 |
| CAPS w/ SuperPoint kp. | 59.3 / 26.1 | **72.9 / 30.5** | **53.5 / 27.9** | **38.1 / 19.2** |

학습은 MegaDepth(야외)만인데 실내 ScanNet에서도 준수 — 일반화 주장의 근거. 단 ScanNet에선 R2D2에 밀린다.

**3D reconstruction (ETH local features benchmark, Table 3)**: Madrid Metropolis(1,344장)에서 CAPS 등록 851장 / sparse point 242K / observation **1,489K**(최다) / track length 6.16 / reproj. err. 1.03px — SIFT(500장, 0.61px), SOSNet(844장, 0.70px) 대비 재구성 완전성(등록·관측 수)은 최고 수준, reprojection error는 열세. 논문도 "매치가 적으면 reproj. err.가 낮아지는 trade-off"로 설명한다. Gendarmenmarkt에선 observation 3,330K로 최다, Tower of London에선 등록 1,104장으로 최다.

**Ablation (MegaDepth ~20K쌍, 10 epochs)**: *Ours*가 같은 GT 대응으로 학습한 *Triplet Loss*보다 좋고(기하 거리 기반 손실 + coarse-to-fine 덕), GT 대응으로 학습한 *Ours supervised*가 최고 — 포즈 supervision과의 격차는 작다. cycle consistency는 개선이 marginal하지만 **cycle만으로는 학습이 실패** — epipolar 항이 본체임을 확인. reweighting 제거와 c2f 제거는 모두 뚜렷한 성능 하락. from scratch(ImageNet pretrain 없이)도 수렴.

## 내 실습 연결

석사 연구(선체 블록 wide-baseline 스테레오 대응 기반 치수 계측)에서 CAPS는 COTR과 함께 **기존 방법 베이스라인**이었다. 포즈만으로 dense descriptor를 학습한다는 설정 자체가 매력적이었지만, 실선박 데이터에서 CAPS 계열의 실패 모드는 이 논문의 3.4절 주장을 뒤집어 읽게 만든다. 논문은 "epipolar 제약이 선 밖의 오답을 대량 억제하고, 선 위의 후보 중엔 진짜 대응이 외형상 가장 닮았을 것"이라는 가정으로 weak supervision의 충분성을 설명하는데, 선체 블록처럼 **동일 구조가 반복되는 표면에서는 그 가정이 정확히 깨진다** — epipolar line이 닮은 반복 구조 여러 개를 관통하면 선 위 후보들의 외형이 서로 구분되지 않고, correlation 분포가 multimodal해지며 기대값(Eq. 5)은 모드 사이 어중간한 위치로 끌려간다. coarse-to-fine의 local window가 이 문제를 줄이려는 장치지만, coarse 최빈값 자체가 엉뚱한 반복 구조에 찍히면 window가 오답 주변만 보게 된다.

이 관찰이 guided mask 접근의 동기가 됐다: epipolar/외형 제약만으로 후보를 줄이는 대신, U-Net 기반 마스크로 **유효 영역 자체를 먼저 제약**해 매칭기가 볼 문맥을 좁힌 뒤 정밀 매칭을 돌리는 파이프라인으로 최종 **29.1 → 2.23px**를 얻었다. CAPS의 uncertainty(분포 분산) 개념은 반대로 유용했다 — 반복 패턴 위의 query는 분산이 커지므로, 실패를 사전에 감지하는 신호로 읽을 수 있다.

## 한 줄 평 / 한계

**한 줄 평.** "supervision을 GT 대응에서 카메라 포즈로 낮춘다"는 문제 설정과, 그걸 가능하게 한 두 장치 — 점–epipolar line 거리 손실과 expectation 기반 미분 가능 매칭 — 의 결합이 깔끔한 논문. soft-argmax 매칭과 coarse-to-fine 구조는 이후 LoFTR 계열 detector-free 매처의 표준 부품이 되는 설계를 선취했고, "geometry가 곧 supervision"이라는 발상은 포즈가 흔한 로보틱스 데이터에 특히 잘 맞는다.

**한계.**

- **epipolar 제약은 1차원 제약일 뿐**: 선 위 후보들 간 구분은 전적으로 외형 유사도에 맡겨진다 — 반복 패턴·저텍스처에서 무너지는 구조적 약점(내 실측의 실패 모드).
- **기대값 매칭의 multimodality 취약성**: 분포가 여러 모드를 가지면 기대값이 모드 사이로 흘러내린다. coarse-to-fine이 완화하지만 coarse 오판까지 구제하진 못한다.
- **detector 비의존이 아니다**: descriptor만 학습하므로 keypoint는 여전히 SIFT/SuperPoint에 의존하고, 성능도 detector 선택에 좌우된다(SIFT kp. vs SuperPoint kp. 격차).
- **실내 일반화의 한도**: ScanNet에선 R2D2 대비 열세 — 야외 photo-tourism(MegaDepth) 학습 분포의 흔적.
- **reprojection error 열세**: SfM 벤치에서 완전성은 얻지만 기하 정밀도 지표에선 SIFT·SOSNet에 밀린다.
