# Mip-Splatting (CVPR 2024)

*3DGS의 줌 아티팩트는 버그가 아니라 샘플링 정리 위반이었다 — 훈련 뷰가 허용하는 최대 주파수로 3D를 band-limit하고, 화면공간 dilation을 물리적 Mip filter로 바꾼다*

Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, Andreas Geiger, *Mip-Splatting: Alias-free 3D Gaussian Splatting*, CVPR 2024 **Best Student Paper**. ([arXiv:2311.16493](https://arxiv.org/abs/2311.16493) · [프로젝트/코드](https://niujinshuchong.github.io/mip-splatting))

## 한 줄 요약

> 3DGS가 줌인/줌아웃에서 무너지는 원인을 **3D frequency 제약의 부재 + 화면공간 2D dilation**으로 특정하고, (1) 훈련 뷰들의 focal length·depth에서 유도한 최대 샘플링 주파수로 각 Gaussian을 컨볼브하는 **3D smoothing filter**, (2) dilation을 물리적 imaging process의 box filter 근사로 교체한 **2D Mip filter** — 단 두 개의 닫힌형 수정으로, single-scale 학습 후 다른 배율 테스트에서 3DGS를 Blender 기준 평균 PSNR **+7.1 dB**(24.84→31.97) 끌어올렸다.

## 문제 — 2D dilation은 어디서 아티팩트를 만드는가

3DGS는 각 Gaussian $\mathcal{G}_k(\mathbf{x})=e^{-\frac{1}{2}(\mathbf{x}-\mathbf{p}_k)^T\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\mathbf{p}_k)}$ (Eq. 1)를 카메라 좌표로 변환하고 local affine Jacobian으로 ray space에 투영해($\boldsymbol{\Sigma}''_k=\mathbf{J}_k\boldsymbol{\Sigma}'_k\mathbf{J}_k^T$, Eq. 3) 2D covariance $\boldsymbol{\Sigma}_k^{2D}$를 얻은 뒤 alpha blending한다. 이때 sub-pixel Gaussian의 degenerate 케이스를 피하려고 화면공간에서 **dilation**을 건다 (Eq. 5):

$$\mathcal{G}^{2D}_k(\mathbf{x})=e^{-\frac{1}{2}(\mathbf{x}-\mathbf{p}_k)^T(\boldsymbol{\Sigma}^{2D}_k+s\mathbf{I})^{-1}(\mathbf{x}-\mathbf{p}_k)}$$

스칼라 $s$만큼 covariance를 부풀리되 **최댓값(중심 opacity)은 그대로 두는** 연산이라, 원문은 morphology의 dilation에 빗대 "2D screen space dilation"이라 부른다(원조 3DGS 논문에는 언급이 없고 코드에만 있다는 각주 포함).

문제는 이 dilation이 최적화와 얽히며 생기는 **scale 모호성**이다. Fig. 1의 사고실험: 5픽셀 센서 앞의 물체를 (a) 충실한 크기의 3D Gaussian으로 표현해도, (b) Dirac $\delta$에 가까운 degenerate Gaussian으로 표현해도 — dilation kernel(크기 ≈ 1픽셀)이 씌워지면 화면에서는 거의 같은 이미지가 나온다. 훈련 뷰만 맞추면 되는 photometric loss는 이 둘을 구분하지 못하고, 실제로 3DGS는 **shrinkage bias** 탓에 Gaussian scale을 체계적으로 과소추정한다. 같은 샘플링 레이트에서는 문제가 안 보이지만:

- **줌인(focal↑ / 접근)**: 투영된 2D Gaussian은 커지는데 dilation은 고정 → 상대적으로 **erosion** — 구조가 실제보다 얇게 그려지고(브레이크 케이블), thin degenerate Gaussian들이 고주파 아티팩트로 드러난다 (Fig. 1d).
- **줌아웃(focal↓ / 후퇴)**: 투영 footprint가 1픽셀보다 작아져도 dilated Gaussian은 감쇠 없이 그대로 → 물리적으로 픽셀에 도달하는 것보다 **더 많은 빛을 누적** — 밝아짐 + 자전거 스포크가 두꺼워지는 dilation 아티팩트 (Fig. 1c).

dilation을 그냥 빼면 되지 않나? 원문 검증: Mip-NeRF 360 같은 복잡 장면에서는 density control이 작은 Gaussian을 대량 생성해 **A100 40GB에서도 OOM**이 나고(보충 §8.1), 학습이 되더라도 anti-aliasing이 없어 줌아웃에서 aliasing이 남는다.

## 방법

### 3D smoothing filter — 샘플링 정리로 3D 표현을 band-limit

핵심 통찰: 재구성 가능한 3D 장면의 최대 주파수는 **훈련 이미지들의 샘플링 레이트가 결정**한다(Nyquist-Shannon: 샘플링 레이트 $\hat\nu$면 $\hat\nu/2$까지만 복원 가능). focal length $f$(픽셀 단위) 이미지에서 화면공간 샘플링 간격 1픽셀을 depth $d$로 back-project하면 world space 샘플링 간격과 주파수는 (Eq. 6):

$$\hat{T}=\frac{1}{\hat\nu}=\frac{d}{f}$$

즉 이 뷰가 복원할 수 있는 주파수 상한은 $\frac{\hat\nu}{2}=\frac{f}{2d}$이고, $2\hat T$보다 작은 primitive는 splatting에서 aliasing을 일으킬 수 있다. depth는 Gaussian 중심 $\mathbf{p}_k$로 근사하고 occlusion은 무시한 채, primitive $k$를 보는 모든 뷰 중 **최대** 샘플링 레이트를 취한다 (Eq. 7):

$$\hat\nu_k=\max\left(\left\{\mathbb{1}_n(\mathbf{p}_k)\cdot\frac{f_n}{d_n}\right\}_{n=1}^{N}\right)$$

$\mathbb{1}_n$은 $\mathbf{p}_k$가 $n$번 카메라 view frustum 안에 있는지의 indicator — "적어도 한 카메라는 이 primitive를 복원할 수 있다"가 기준이다. 이 $\hat\nu_k$로 각 Gaussian에 저역 Gaussian filter $\mathcal{G}_{\text{low}}$를 **projection 전에 3D에서** 컨볼브한다 (Eq. 8):

$$\mathcal{G}_k(\mathbf{x})_{\text{reg}}=(\mathcal{G}_k\otimes\mathcal{G}_{\text{low}})(\mathbf{x})$$

Gaussian끼리의 convolution은 다시 Gaussian이고 covariance가 더해지는 성질($\boldsymbol{\Sigma}_1+\boldsymbol{\Sigma}_2$) 덕에 닫힌형이 된다 (Eq. 9):

$$\mathcal{G}_k(\mathbf{x})_{\text{reg}}=\sqrt{\frac{|\boldsymbol{\Sigma}_k|}{|\boldsymbol{\Sigma}_k+\frac{s}{\hat\nu_k}\cdot\mathbf{I}|}}\;e^{-\frac{1}{2}(\mathbf{x}-\mathbf{p}_k)^T(\boldsymbol{\Sigma}_k+\frac{s}{\hat\nu_k}\cdot\mathbf{I})^{-1}(\mathbf{x}-\mathbf{p}_k)}$$

앞의 제곱근 계수가 convolution의 **에너지 보존**(dilation과 달리 부풀린 만큼 최댓값을 낮춤)이고, filter scale $\frac{s}{\hat\nu_k}$는 primitive마다 다르다 — 가까이서/긴 focal로 관측된 Gaussian일수록 filter가 작다. 어떤 Gaussian도 자기 최대 샘플링 레이트의 절반을 넘는 주파수를 못 갖게 되어 **줌인 시 고주파 아티팩트가 사라지고**, 학습 후 $\mathcal{G}_{\text{low}}$는 표현에 융합되어(Eq. 9 그대로) 시점과 무관한 **장면의 고유 속성**이 된다.

### 2D Mip filter — dilation을 물리적 box filter 근사로 교체

3D filter만으로는 줌아웃(더 낮은 샘플링 레이트) 시 aliasing이 남는다. 물리적 imaging process에서 픽셀에 닿는 photon은 **픽셀 면적에 걸쳐 적분**된다 — 이상적으로는 image space의 2D box filter. 이를 효율을 위해 2D Gaussian으로 근사해 dilation을 대체한다 (Eq. 10):

$$\mathcal{G}^{2D}_k(\mathbf{x})_{\text{mip}}=\sqrt{\frac{|\boldsymbol{\Sigma}^{2D}_k|}{|\boldsymbol{\Sigma}^{2D}_k+s\mathbf{I}|}}\;e^{-\frac{1}{2}(\mathbf{x}-\mathbf{p}_k)^T(\boldsymbol{\Sigma}^{2D}_k+s\mathbf{I})^{-1}(\mathbf{x}-\mathbf{p}_k)}$$

Eq. 5와의 유일한 차이가 바로 정규화 계수 $\sqrt{|\boldsymbol{\Sigma}^{2D}_k|/|\boldsymbol{\Sigma}^{2D}_k+s\mathbf{I}|}$인데, 이것이 결정적이다: sub-pixel Gaussian이 부풀 때 그만큼 opacity가 깎여 **줌아웃에서 빛이 과누적되지 않는다**. EWA splatting과 형태는 비슷하지만 원리가 다르다는 점을 원문이 강조한다 — Mip filter는 **정확히 1픽셀**의 box filter를 재현하는 게 목표고 $s$는 픽셀 하나를 덮게 고정, EWA는 렌더 이미지의 bandwidth 제한이 목표라 filter 크기를 경험적으로 잡으며(EWA 논문은 identity covariance = 3×3픽셀 영역까지 권장) 줌아웃에서 과도하게 뭉개진다.

### 구현 디테일 (원문 §6.1, §6.4)

- 공개 3DGS 코드베이스 위에 구현, **30K iterations**, loss·density control·스케줄·하이퍼파라미터는 3DGS와 동일.
- 2D Mip filter variance **0.1**(1픽셀 근사) + 3D smoothing filter variance **0.2** = 합 **0.3** — dilation 0.3을 쓰는 3DGS/3DGS+EWA와의 공정 비교를 위한 배분.
- $\hat\nu_k$는 매 **$m=100$ iterations**마다 재계산(중심이 학습 내내 안정적이라 충분). 현재 PyTorch 구현이라 약간의 학습 오버헤드 — CUDA화가 future work.

## 결과 — 원문 수치

**Blender, multi-scale 학습/테스트** (Table 1, 40% full-res + 각 저해상도 20%씩 샘플링, PSNR/SSIM/LPIPS 4-scale 평균):

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| MipNeRF | 34.51 | 0.973 | 0.026 |
| Tri-MipRF* | 34.36 | 0.974 | 0.026 |
| 3DGS | 29.77 | 0.960 | 0.040 |
| 3DGS + EWA | 33.01 | 0.974 | 0.027 |
| **Mip-Splatting** | **34.56** | **0.979** | **0.019** |

**Blender, single-scale(full-res) 학습 → 1, ½, ¼, ⅛ 해상도 테스트** (Table 2, 줌아웃 시뮬레이션) — 이 논문이 새로 제안한 out-of-distribution 프로토콜:

| 방법 | Full | ½ | ¼ | ⅛ | Avg |
|---|---|---|---|---|---|
| MipNeRF | 33.08 | 33.31 | 30.91 | 27.97 | 31.31 |
| 3DGS | **33.33** | 26.95 | 21.38 | 17.69 | 24.84 |
| 3DGS + EWA | 33.51 | 31.66 | 27.82 | 24.63 | 29.40 |
| **Mip-Splatting** | 33.36 | **34.00** | **31.85** | **28.67** | **31.97** |

학습 배율에서는 전 방법이 비슷하지만 배율이 멀어질수록 3DGS는 급락(⅛에서 −15.7 dB 차이), EWA는 oversmoothing으로 손해를 보고, Mip-Splatting만 유지된다.

**Mip-NeRF 360, ⅛ 스케일 학습 → 1×·2×·4×·8× 렌더** (Table 3, 줌인 시뮬레이션, PSNR):

| 방법 | 1× | 2× | 4× | 8× | Avg |
|---|---|---|---|---|---|
| mip-NeRF 360 | 29.26 | 25.18 | 24.16 | 24.10 | 25.67 |
| zip-NeRF | **29.66** | 23.27 | 20.87 | 20.27 | 23.52 |
| 3DGS | 29.19 | 23.50 | 20.71 | 19.59 | 23.25 |
| 3DGS + EWA | 29.30 | 25.90 | 23.70 | 22.81 | 25.43 |
| **Mip-Splatting** | 29.39 | **27.39** | **26.47** | **26.22** | **27.37** |

MLP 기반(mip-NeRF 360, zip-NeRF)조차 out-of-distribution 주파수 외삽에 실패하는 반면 Mip-Splatting은 고주파 아티팩트 없는 렌더를 낸다. **표준 same-scale 세팅**(Table 4)에서도 27.79/0.827/0.203으로 3DGS(재학습 27.70)·3DGS+EWA(27.77)와 동급 — 즉 공짜 수정이다.

**Ablation**: 360 줌인에서 3D smoothing filter 제거 시 평균 PSNR 27.37→26.93 + 고주파 아티팩트(Table 5), Blender 줌아웃에서 2D Mip filter 제거 시 31.97→30.76에 저해상도로 갈수록 급락(Table 6) — 각 filter의 역할 분담(3D=줌인, 2D Mip=줌아웃)이 수치로 확인된다.

## 내 실습 연결

리콘랩스에서 3DGS 프로덕션 품질 연구를 하며 **mip-splatting을 직접 재현해 파이프라인 baseline으로 삼았다**. 이유가 정확히 이 논문의 문제 설정과 겹친다 — 제품 촬영 데이터는 클로즈업과 풀샷이 섞여 들어오고, 사용자는 뷰어에서 자유롭게 줌한다. 즉 "single-scale 학습 후 다른 배율 렌더"가 실험 프로토콜이 아니라 **일상 운영 조건**이다. 바닐라 3DGS로는 줌인 시 얇은 구조(제품 스트랩·와이어)가 갈라지고 줌아웃 시 밝아지는 현상을 실데이터에서 그대로 재확인했고, mip-splatting의 3D smoothing filter가 shrinkage bias로 생기는 needle-like Gaussian을 눌러주는 것이 floater/스파이크 아티팩트 억제에도 유효했다. 이후 geometry 품질(depth·mesh 추출)을 위해 RaDe-GS를 채택·통합할 때도 이 baseline과의 스케일-강건성 비교가 판단 기준이었다. 석사 때 다룬 camera calibration 관점에서도 Eq. 6-7이 반갑다 — $f/d$로 샘플링 한계를 잡는 건 결국 픽셀의 IFOV를 world space로 편 것이라, 계측에서 쓰던 공간 분해능 계산과 같은 문법이다.

## 한 줄 평 / 한계

**평**: "3DGS의 아티팩트는 표현의 한계가 아니라 주파수 제약의 부재"라는 진단이 정확했고, 해법이 학습 스케줄도 아키텍처도 아닌 **닫힌형 filter 두 개**(코드 수 줄)라는 점이 Best Student Paper다운 우아함이다. Gaussian convolution의 대수적 성질을 정확히 골라 쓴, geometry 기반 논문의 모범.

**한계(원문 §6.4 자인)**: box filter의 Gaussian 근사 오차는 화면상 Gaussian이 작을수록 커져 **깊은 줌아웃에서 오차가 증가**하고(Table 2 경향과 일치), $\hat\nu_k$ 재계산이 PyTorch라 학습 오버헤드가 있다. 추가로 — occlusion을 무시한 $\max$ 샘플링 레이트는 가려진 primitive에 과대한 주파수 허용치를 줄 수 있고, 학습 뷰 분포가 편향된 실촬영(제품 근접샷 위주)에서는 $\hat\nu_k$가 장면 일부에서만 과도하게 커져 filter가 사실상 꺼지는 영역이 생긴다 — 실무에서는 뷰 커버리지 설계가 여전히 품질의 절반이다.
