# RaDe-GS (2024)

*3D 가우시안을 납작하게 만들지 않고도 — ray별 최대 기여점을 닫힌형으로 유도해, 래스터라이저 안에서 정확한 depth·normal을 그냥 "그려낸다"*

Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, Ping Tan, *RaDe-GS: Rasterizing Depth in Gaussian Splatting*, HKUST + Simon Fraser Univ., 2024. ([arXiv:2406.01467](https://arxiv.org/abs/2406.01467), [프로젝트](https://baowenz.github.io/radegs/))

## 한 줄 요약

> 3DGS의 깊이는 "가우시안 중심 하나 = 스플랫 전체의 깊이"라 형상을 못 담는다. RaDe-GS는 **각 광선에서 가우시안 값이 최대가 되는 지점을 폐형해로 유도**하고, 근사 아핀 투영 아래서 이 교점들이 **ray space의 한 평면 위에 놓임**을 보여 — 스플랫별 깊이가 화면에서 `중심 깊이 + 선형 보정항`이라는 평면식으로 래스터라이즈된다. 그 결과 **3D 가우시안 표현력을 그대로 유지한 채** DTU Chamfer 0.68 mm(2DGS 0.80, GOF 0.74, 3DGS 1.96)로 implicit 강자 NeuralAngelo(0.61 mm)에 근접하면서, 학습은 8.3분(NeuralAngelo >12h)이다.

## 문제

3DGS는 분·초 단위 학습과 실시간 렌더링을 주지만, 이산·비구조적인 반투명 가우시안 집합에서 **표면을 뽑는 것**이 어렵다. 원문이 짚는 기존 접근의 대가:

- **3DGS의 depth**: α-블렌딩 정렬용으로 각 가우시안의 **중심 깊이 하나**를 쓴다. 스플랫 내부에서 깊이가 상수라 형상 디테일을 못 담고, 메시가 노이즈로 뒤덮인다(DTU에서 CD 1.96 mm).
- **2DGS·SuGaR — 납작하게 만들기**: 가우시안을 평면 디스크로 강제하면 표면 추출은 쉬워지지만, **저차원 표현이 최적화를 불안정하게** 만든다. 풀·나뭇가지 같은 복잡한 형상에서 degenerate하고, novel view PSNR도 떨어진다(원문 Table 3·4에서 2DGS·SuGaR가 3DGS보다 낮음 — "planar constraint가 복잡한 장면에서 모델 성능을 해친다"는 것이 원문 진단).
- **GOF — ray tracing**: 광선을 따라 opacity를 적분해 정확한 표면을 얻지만, DTU 한 장면에 **약 55분~1시간**(표준 GS는 5.2분).

즉 기존 선택지는 "표현력을 깎거나(2DGS), 시간을 태우거나(GOF)"였다. RaDe-GS의 질문: **일반 3D 가우시안을 건드리지 않고, 래스터라이제이션 비용 그대로 정확한 depth를 계산할 수 없나?**

## 방법

### 배경 — 3DGS 파이프라인

3D 가우시안 $G(\mathbf{x})=e^{-(\mathbf{x}-\mathbf{x}_c)^\top\Sigma^{-1}(\mathbf{x}-\mathbf{x}_c)}$, $\Sigma=\mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$. 원근 투영을 가우시안마다 국소 아핀 변환으로 근사(Zwicker EWA)해 $\Sigma'=\mathbf{J}\mathbf{W}\Sigma\mathbf{W}^\top\mathbf{J}^\top$로 2D 공분산을 얻고, 깊이순 α-블렌딩 $c=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$로 렌더링한다. 여기까지가 표준 — RaDe-GS는 이 위에 depth·normal 래스터라이저만 얹는다.

### 핵심 1 — 광선-가우시안 교점의 폐형해 (camera space)

광선 $\mathbf{x}=\mathbf{o}+t\mathbf{v}$를 따라 가우시안 값은 $t$의 1D 가우시안 함수 $G^1(t)$가 된다. '교점'을 **이 값이 최대가 되는 지점**으로 정의하면 최대점은 닫힌형으로:

$$t^*=\frac{\mathbf{v}^\top\Sigma^{-1}(\mathbf{x}_c-\mathbf{o})}{\mathbf{v}^\top\Sigma^{-1}\mathbf{v}}\qquad(원문\ 식 7)$$

픽셀마다 $\mathbf{v}$가 달라 이 교점들의 집합은 곡면(curved surface)이다 — 그대로는 래스터라이즈하기 비싸다.

### 핵심 2 — ray space에선 이 곡면이 평면이 된다

국소 아핀 투영을 가정하고 Zwicker의 ray space로 옮긴다. camera space의 점 $\mathbf{x}=(x,y,z)^\top$이 $\mathbf{u}=(u,v,t)^\top$로 변환되는데, $(u,v)$는 이미지 평면 좌표, $t=\sqrt{x^2+y^2+z^2}$는 카메라 중심까지의 거리다. 이 좌표계의 요점 두 가지:

- 한 광선 위의 점들은 $(u,v)$를 공유하고 $t$만 다르다 — 광선이 좌표축과 정렬된다.
- 따라서 **광선 방향이 모든 픽셀에서 상수 $\mathbf{v}'=(0,0,1)^\top$로 정규화**된다. camera space에서 픽셀마다 달랐던 $\mathbf{v}$가 사라지는 것이 유도 전체의 지렛대다.

가우시안도 ray space로 변환되어 $G'(\mathbf{u})=e^{-(\mathbf{u}-\mathbf{u}_c)^\top\Sigma'^{-1}(\mathbf{u}-\mathbf{u}_c)}$ (식 8), 중심은 $\mathbf{u}_c=(u_c,v_c,t_c)^\top$. 광선 $\mathbf{u}=\mathbf{u}_o+t\mathbf{v}'$ ($\mathbf{u}_o=(u,v,0)^\top$)을 대입하면 식 7과 같은 꼴의 최대점이

$$t^*=\hat{\mathbf{q}}(\mathbf{u}_c-\mathbf{u}_o),\qquad \hat{\mathbf{q}}=\frac{\mathbf{v}'^\top\Sigma'^{-1}}{\mathbf{v}'^\top\Sigma'^{-1}\mathbf{v}'}\qquad(식 12–13)$$

이 되는데, $\mathbf{v}'$가 상수라 $\hat{\mathbf{q}}$는 **가우시안당 한 번만 미리 계산**하면 된다. 깊이는 $d=\cos\theta\, t^*$이고, $\theta$를 가우시안 중심의 각 $\theta_c$로 근사($\cos\theta_c = z_c/t_c$)하면 $d=\hat{\mathbf{p}}(\mathbf{u}_c-\mathbf{u}_o)$, $\hat{\mathbf{p}}=\frac{z_c}{t_c}\hat{\mathbf{q}}$. 이를 전개하면(식 16–17, $\hat{\mathbf{p}}(0,0,t_c)^\top=z_c$ 증명이 유도의 백미):

$$d=z_c+\mathbf{p}\begin{pmatrix}\Delta u\\ \Delta v\end{pmatrix}\qquad(식 4)$$

$z_c$는 중심 깊이, $\Delta u=u_c-u,\ \Delta v=v_c-v$는 픽셀의 상대 위치, $\mathbf{p}\in\mathbb{R}^{1\times2}$는 가우시안 파라미터와 카메라 외부 파라미터로 정해지는 상수 벡터. 즉 **투영된 가우시안 내부에서 깊이는 "중심 깊이 + 픽셀별 선형 보정항"이라는 평면식** — 3DGS의 상수 깊이에서 정확히 한 차수 올라간, 그러나 래스터라이저가 공짜로 평가할 수 있는 식이다. 최종 depth map은 투영된 가우시안들의 반투명도를 고려한 **median depth**로 합성한다.

### 핵심 3 — 같은 평면에서 normal이 나온다

교점들이 ray space에서 평면 $(\mathbf{q}\ \ 1)(\mathbf{u}-\mathbf{u}_c)=0$ 위에 놓이므로(식 18–20), 법선은 $\mathbf{n}'=-(\mathbf{q}\ \ 1)^\top$(식 21), camera space로는 아핀 행렬로 $\mathbf{n}=\mathbf{J}^\top\mathbf{n}'$(식 22) 후 단위화. depth와 normal이 **같은 유도에서 일관되게** 나온다.

### 정규화와 학습 — 2DGS 손실을 그대로 결합

미분 가능한 depth·normal이 생겼으니 2DGS의 기하 정규화를 그대로 쓸 수 있다:

- **Depth distortion** $\mathcal{L}_d=\sum_{i,j}\omega_i\omega_j(d_i-d_j)^2$ — 한 광선 위 가우시안들의 깊이를 모은다($\omega_i=\alpha_i\prod_{j<i}(1-\alpha_j)$, distortion 계산 시 $\omega$의 gradient는 detach).
- **Normal consistency** $\mathcal{L}_n=\sum_i\omega_i(1-\mathbf{n}_i^\top\tilde{\mathbf{n}})$ — 가우시안 normal과 depth map 유한차분 normal을 일치시킨다.
- 총 손실 $\mathcal{L}=\mathcal{L}_c+w_d\mathcal{L}_d+w_n\mathcal{L}_n$, $w_d=100,\ w_n=5$ (GOF 관행).

**수정 범위** — 3DGS 공개 코드 위에 depth·normal·정규화용 **커스텀 CUDA 커널**을 추가한 것이 전부다:

- Mip-Splatting의 3D filter(anti-aliasing)와 GOF의 densification을 채택, DTU/TNT에선 VastGaussian의 decoupled appearance modeling 결합.
- 학습 스케줄: 15k iteration은 렌더링 손실만 → 이후 15k에 depth/normal 제약을 켠다. NVIDIA H800 1장.
- 메시 추출: 학습 뷰 전체의 depth map을 렌더해 TSDF fusion(Curless–Levoy) + Marching Cubes.
- 초기화: COLMAP sparse point cloud (DTU·TNT 제공 카메라 포즈 사용).

**가우시안 자체는 일반 3D 가우시안 그대로** — 이것이 "표현력을 깎지 않고 기하를 얻는다"는 포지셔닝의 실체이고, local affine 가정은 표준 3DGS가 이미 쓰는 가정이므로 기존 어떤 GS 계열 방법에도 바로 이식 가능하다는 주장의 근거다.

## 결과 (원문 수치)

**DTU Chamfer Distance (mm, 15장면 평균, half-res)** — Table 1:

| | 3DGS | SuGaR | 2DGS | GOF | **RaDe-GS (30K)** | NeuralAngelo |
|---|---|---|---|---|---|---|
| CD | 1.96 | 1.33 | 0.80 | 0.74 | **0.68** | 0.61 |
| 시간 | 5.2m | 52m | 8.9m | 55m | **8.3m** | >12h |

20K iteration이면 **5.0분에 0.69 mm** — 3DGS의 학습 시간으로 explicit 최고 정확도. Full-res에서도 0.69 mm/20.2분(2DGS 0.87/26.6m, GOF 0.82/136m).

**TNT F1 (6장면 평균)** — Table 2: RaDe-GS **0.37**(TSDF, 11.5분), marching tetrahedra 사용 시 0.40. 2DGS 0.30(12.3m), GOF 0.34/♯0.46(69m), NeuS 0.38, NeuralAngelo 0.50(>24h). TSDF+GS 조합 중 최고지만 NeuralAngelo엔 미달 — 원문은 대규모 장면에서 저해상도 voxel TSDF가 병목이라 진단.

**Mip-NeRF360 NVS** — Table 3: 실외 PSNR **25.17**/SSIM 0.764/LPIPS **0.199**(3DGS 24.64/0.731/0.234, 2DGS 24.34, GOF 24.82, Mip-Splatting 24.65), 실내 30.74/0.928/LPIPS **0.165**. 기하 정규화를 켜고도 3DGS보다 나은 NVS — 2DGS·SuGaR가 3DGS보다 떨어지는 것과 대조되며, 원문은 이를 "planar 제약이 복잡 장면에서 모델을 해치는 반면, 우리는 원래의 3D 가우시안을 유지해 더 나은 데이터 표현을 얻는다"로 해석한다. **Synthetic NeRF PSNR 평균 33.60**(GOF 33.46, 3DGS 33.32, 2DGS 33.07, 정규화 켠 2DGS★ 32.65) — Table 4.

**계산 효율 (§4.2.3)**: 픽셀별 기하 정규화의 forward/backward가 더해져 3DGS(5.2m)보다 약간 느린 8.3분이지만, 2DGS(8.9m)보다 빠르고 GOF(55m~1h)의 ray tracing 대비 자릿수가 다르다. 더 적은 iteration(20k)에서 3DGS보다 높은 정확도와 효율을 동시에 달성 — depth·normal 제약이 최적화를 오히려 가속하는 방향으로 작동한다.

정성 비교(Fig. 5·7)에서 3DGS는 부정확한 depth 근사로 노이즈 낀 메시를, 2DGS·SuGaR는 specular highlight 영역에서 불안정한 표면을 만드는 반면 RaDe-GS는 매끈하고 정밀한 형상을 유지한다. Fig. 1의 풀·그루터기 장면에서는 2DGS가 blurry 렌더링과 노이즈 형상을 내 기본 설정으로 mesh 추출 전 필터링을 쓴다는 점도 지적된다.

## 내 실습 연결

리콘랩스 3DGS 프로덕션 품질 개선에서 **mip-splatting 재현을 Baseline으로 두고 RaDe-GS를 최종 채택**해 실서비스 파이프라인에 통합했다(Dockerfile/kubeflow 이미지, CUDA 커널 재빌드). 채택 논리가 정확히 이 논문의 포지셔닝과 일치한다:

- **3D 가우시안 유지 = 배경 표현력**: 실촬영 데이터 비교에서 Baseline은 배경이 검은 공동으로 붕괴했지만, RaDe-GS 기반 Improved는 벽·그림까지 복원됐다. 2DGS 계열이었다면 원문 Fig. 1의 풀·나뭇가지처럼 복잡 영역에서 degenerate했을 것 — "planar constraint가 복잡 장면을 해친다"는 원문 진단을 서비스 데이터로 재확인한 셈.
- **Baseline이 논문 안에 있다**: RaDe-GS는 Mip-Splatting의 3D filter를 내장한다. 내가 재현한 Baseline이 채택 모델의 anti-aliasing 구성요소로 그대로 들어가 있어, 마이그레이션이 "교체"가 아니라 "상위호환"이었다.
- **정확한 depth = 치수측정의 기반**: 복원 결과에서 h·w·d를 재는 상품 치수측정 데모는 depth가 스플랫 중심값이 아니라 평면식으로 정밀해야 성립한다. 래스터라이즈 비용 그대로라 프로덕션 처리 시간 예산도 지켰다(DTU 기준 3DGS 5.2분 vs RaDe-GS 5.0~8.3분).
- 캘리브레이션 스터디와의 접점: 유도 전체가 Zwicker의 local affine 근사 위에 서 있다 — 카메라 기하(원근→아핀 국소화)를 정확히 이해해야 이 논문의 "왜 ray space에선 평면인가"가 읽힌다.

## 한 줄 평 / 한계

**"표현을 바꾸지 말고 렌더링 공식을 바꿔라"** — 2DGS가 표현을 깎아 얻은 것을, RaDe-GS는 유도 하나(ray별 최대점의 폐형해 + ray space 평면성)로 공짜에 가깝게 얻었다. 원문이 명시한 한계: (1) TSDF fusion의 voxel 해상도 제약으로 대규모 장면(Courthouse·Meetingroom)에서 F1 손해 — GOF♯(0.46)에 밀리는 이유이고, multi-resolution TSDF는 future work. (2) 반사 표면 실패 — DTU 금속 가위(scan 110)는 3DGS의 단순 color function 한계 그대로이며 GaussianShader류 결합을 제안. (3) 깊이의 $\cos\theta\approx\cos\theta_c$ 근사와 local affine 근사가 유도 전체의 전제 — 광각·근접에서의 오차는 정량화되지 않았다. 프로덕션 관점에서 하나 더: median depth 합성은 반투명 물체에서 "어느 표면"인지의 정의 문제를 남긴다.
