# 2DGS (SIGGRAPH 2024)

*"볼륨을 눌러 디스크로" — 3D 가우시안의 multi-view 불일치를 primitive 차원에서 제거하고, ray-splat 교차 + 정규화 2개로 GS를 렌더링 도구에서 기하 도구로 바꾼 논문. 원문 PDF 전 페이지를 직접 읽고 정리했다.*

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao, *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*, SIGGRAPH Conference Papers 2024. ([arXiv:2403.17888](https://arxiv.org/abs/2403.17888) · [프로젝트](https://surfsplatting.github.io))

## 한 줄 요약

> 3D 가우시안 볼륨을 **2D oriented disk(surfel)** 로 붕괴시키고, 호모지니어스 좌표의 **두 비평행 평면 교차**로 perspective-정확한 ray-splat 교차를 닫힌형으로 풀며, **depth distortion + normal consistency** 두 정규화로 스플랫을 표면에 정렬 — DTU Chamfer **0.80**(3DGS 1.96, SuGaR 1.33 대비)을 SDF 계열보다 **~100× 빠른 10.9분**에 달성한 표면 지향 Gaussian Splatting.

## 문제 — 3DGS는 왜 기하가 부정확한가

3DGS는 3D 공분산 $\Sigma=\mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$의 볼륨 가우시안 $\mathcal{G}(\mathbf{p})=\exp(-\tfrac12(\mathbf{p}-\mathbf{p}_k)^\top\Sigma^{-1}(\mathbf{p}-\mathbf{p}_k))$을 affine 근사 $\Sigma'=\mathbf{J}\mathbf{W}\Sigma\mathbf{W}^\top\mathbf{J}^\top$로 투영해 알파블렌딩한다. 표면 복원 관점에서 원문이 짚는 결함은 네 가지다.

1. **볼륨 radiance vs 얇은 표면의 충돌** — 각도 전체의 radiance를 blob 하나에 담는 표현이 표면의 얇음과 어긋난다.
2. **노멀 부재** — 표면 복원에 필수인 노멀을 3DGS는 원래 모델링하지 않는다.
3. **multi-view 불일치** — 3DGS는 픽셀 레이와 3D 가우시안의 교차점에서 값을 평가하는데, **보는 방향마다 다른 2D 교차 단면(intersection plane)** 을 자르게 된다(Fig. 2). 같은 primitive가 뷰마다 다른 깊이/값을 내놓으니 depth를 융합하면 노이즈가 된다.
4. **affine 투영 근사** — 가우시안 중심 근처에서만 정확하고 주변부에서 perspective 오차가 커진다(Zwicker et al. 2004).

2DGS의 답: primitive 자체를 평면 디스크로 만들면 교차 단면이 뷰와 무관하게 그 평면 자체가 되고(뷰 일관), 노멀은 디스크의 법선으로 공짜로 나온다.

## 방법

### 2D oriented disk (surfel) 정의

스플랫 하나는 중심 $\mathbf{p}_k$, 두 직교 tangential 벡터 $\mathbf{t}_u,\mathbf{t}_v$, 스케일 $\mathbf{S}=(s_u,s_v)$로 정의된다. 법선은 $\mathbf{t}_w=\mathbf{t}_u\times\mathbf{t}_v$, 회전은 $\mathbf{R}=[\mathbf{t}_u,\mathbf{t}_v,\mathbf{t}_w]$, 스케일 행렬은 마지막 원소가 0인 대각행렬 — 즉 한 축을 눌러버린 "납작한" 가우시안이다. 로컬 tangent 평면의 파라미터화:

$$P(u,v)=\mathbf{p}_k+s_u\mathbf{t}_u u+s_v\mathbf{t}_v v=\mathbf{H}(u,v,1,1)^\top,\qquad \mathbf{H}=\begin{bmatrix}s_u\mathbf{t}_u & s_v\mathbf{t}_v & \mathbf{0} & \mathbf{p}_k\\ 0&0&0&1\end{bmatrix}=\begin{bmatrix}\mathbf{RS} & \mathbf{p}_k\\ \mathbf{0} & 1\end{bmatrix}$$

$uv$ 공간에서의 값은 표준 가우시안 $\mathcal{G}(\mathbf{u})=\exp\!\left(-\tfrac{u^2+v^2}{2}\right)$. 학습 파라미터는 $\mathbf{p}_k,(s_u,s_v),(\mathbf{t}_u,\mathbf{t}_v)$ + 3DGS와 동일한 opacity $\alpha$, SH 색.

### Ray-splat 교차 — 두 비평행 평면의 교차

affine 근사 대신 명시적 교차를 쓴다. 핵심 트릭(Sigg et al. 2006, Weyrich et al. 2007 유래): 픽셀 $\mathbf{x}=(x,y)$의 레이를 **두 직교 평면의 교차**로 표현한다. x-plane은 4D 호모지니어스 평면 $\mathbf{h}_x=(-1,0,0,x)^\top$, y-plane은 $\mathbf{h}_y=(0,-1,0,y)^\top$. world→screen 변환 $\mathbf{W}\in4\times4$에 대해 스크린 점은 $\mathbf{x}=(xz,yz,z,z)^\top=\mathbf{W}P(u,v)=\mathbf{W}\mathbf{H}(u,v,1,1)^\top$.

평면은 점과 반대로 역전치로 변환된다($\mathbf{M}$으로 점을 옮기는 것 = $\mathbf{M}^{-\top}$로 평면 파라미터를 옮기는 것, Blinn 1977). $\mathbf{M}=(\mathbf{WH})^{-1}$을 적용해야 할 자리에 $(\mathbf{WH})^\top$를 쓰면 되므로 **명시적 역행렬이 사라진다**:

$$\mathbf{h}_u=(\mathbf{W}\mathbf{H})^\top\mathbf{h}_x,\qquad \mathbf{h}_v=(\mathbf{W}\mathbf{H})^\top\mathbf{h}_y$$

교차점 $(u,v,1,1)$은 두 평면 위에 동시에 있어야 하므로 $\mathbf{h}_u\cdot(u,v,1,1)^\top=\mathbf{h}_v\cdot(u,v,1,1)^\top=0$, 이 2×2 선형계의 닫힌 해:

$$u(\mathbf{x})=\frac{\mathbf{h}_u^2\mathbf{h}_v^4-\mathbf{h}_u^4\mathbf{h}_v^2}{\mathbf{h}_u^1\mathbf{h}_v^2-\mathbf{h}_u^2\mathbf{h}_v^1},\qquad v(\mathbf{x})=\frac{\mathbf{h}_u^4\mathbf{h}_v^1-\mathbf{h}_u^1\mathbf{h}_v^4}{\mathbf{h}_u^1\mathbf{h}_v^2-\mathbf{h}_u^2\mathbf{h}_v^1}$$

($\mathbf{h}^i$는 $i$번째 성분, $\mathbf{h}_u^3=\mathbf{h}_v^3=0$은 $\mathbf{H}$ 구조상 자동). 이 $(u,v)$로 가우시안 값을 평가하고 깊이 $z$도 같은 식에서 얻는다 — 픽셀마다 **perspective-정확한** 스플래팅이며, 전체가 미분 가능해서 역전파가 그대로 통한다. Zwicker 2001a의 호모그래피 기반 정식화를 계승하되, conic 투영의 수치 불안정(옆에서 본 스플랫이 선분으로 퇴화 시 역변환 폭발)과 임계값 폐기가 없다.

### Degenerate 케이스 — object-space low-pass filter

비스듬히 본 2D 가우시안은 스크린에서 선분으로 퇴화해 래스터화에서 누락될 수 있다. Botsch et al. 2005의 object-space low-pass filter로 안정화:

$$\hat{\mathcal{G}}(\mathbf{x})=\max\left\{\mathcal{G}(\mathbf{u}(\mathbf{x})),\,\mathcal{G}\!\left(\frac{\mathbf{x}-\mathbf{c}}{\sigma}\right)\right\}$$

$\mathbf{c}$는 중심 $\mathbf{p}_k$의 투영, $\sigma=\sqrt{2}/2$ — 화면상 최소 반경 $\sigma$의 가우시안으로 하한을 걸어 렌더링에 항상 충분한 픽셀이 잡히게 한다. 래스터화는 3DGS와 동일(바운딩 박스 → 깊이 정렬 → 타일 → front-to-back 알파블렌딩 $\mathbf{c}(\mathbf{x})=\sum_i\mathbf{c}_i\alpha_i\hat{\mathcal{G}}_i\prod_{j<i}(1-\alpha_j\hat{\mathcal{G}}_j)$).

### 정규화 1 — Depth Distortion

photometric loss만으로는 가우시안이 레이를 따라 퍼져 있어도 색은 비슷하게 나온다(볼륨 렌더링은 교차 간 거리를 안 본다). Mip-NeRF360의 distortion loss에서 착안하되, **교차 깊이 $z_i$를 직접 조정**하도록 만든다:

$$\mathcal{L}_d=\sum_{i,j}\omega_i\omega_j|z_i-z_j|,\qquad \omega_i=\alpha_i\hat{\mathcal{G}}_i(\mathbf{u}(\mathbf{x}))\prod_{j=1}^{i-1}(1-\alpha_j\hat{\mathcal{G}}_j(\mathbf{u}(\mathbf{x})))$$

레이 위 교차들의 blending weight 쌍별 깊이 차이를 벌줘서 스플랫들을 얇은 구간으로 집중시킨다. Mip-NeRF360과 달리 $z_i$가 샘플 간격(비최적화 대상)이 아니라 ray-splat 교차 깊이라서 그래디언트가 primitive를 직접 밀착시킨다. 구현은 부록 A: 깊이를 NDC로 변환(near/far 0.2/1000)한 $\mathcal{L}_2$형을 누적량 $A_i=\sum\omega_j,\ D_i=\sum\omega_j m_j,\ D_i^2=\sum\omega_j m_j^2$로 풀어 **단일 forward pass**로 CUDA 구현(Sun et al. 2022b식).

### 정규화 2 — Normal Consistency

레이 위 반투명 surfel이 여럿일 수 있으므로 누적 opacity가 0.5가 되는 **median 교차점 $\mathbf{p}_s$** 를 실제 표면으로 보고, 스플랫 노멀과 깊이맵 그라디언트 노멀을 정렬한다:

$$\mathcal{L}_n=\sum_i\omega_i(1-\mathbf{n}_i^\top\mathbf{N}),\qquad \mathbf{N}(x,y)=\frac{\nabla_x\mathbf{p}_s\times\nabla_y\mathbf{p}_s}{|\nabla_x\mathbf{p}_s\times\nabla_y\mathbf{p}_s|}$$

$\mathbf{n}_i$는 카메라를 향한 스플랫 노멀, $\mathbf{N}$은 인접 깊이 점들의 유한차분. "primitive가 주장하는 기하(노멀)"와 "렌더링이 만드는 기하(깊이의 미분)"를 서로 맞추는 자기일관 제약이다.

최종 손실 $\mathcal{L}=\mathcal{L}_c+\alpha\mathcal{L}_d+\beta\mathcal{L}_n$ — $\mathcal{L}_c$는 3DGS와 같은 $\mathcal{L}_1$+D-SSIM, $\alpha=1000$(bounded)/$100$(unbounded), $\beta=0.05$.

### 메시 추출 — TSDF 융합

학습 뷰들의 깊이맵(median depth $z_{\text{median}}=\max\{z_i\,|\,T_i>0.5\}$; 부록 B — mean depth $z_{\text{mean}}=\sum_i\omega_i z_i/(\sum_i\omega_i+\epsilon)$보다 outlier에 강건)을 렌더해 Open3D의 truncated signed distance fusion(TSDF)으로 융합한다. voxel size 0.004, truncation threshold 0.02. 학습 자체는 3DGS의 adaptive densification을 따르되 gradient threshold 0.0002, opacity<0.05 스플랫을 3000 step마다 제거, 전부 RTX 3090 한 장.

## 결과 — 원문 수치

**DTU Chamfer distance (mm, Table 1·3)** — 15 scene 평균:

| 방법 | Chamfer ↓ | 시간 | 저장 |
|---|---|---|---|
| NeRF | 1.49 | >12h | — |
| VolSDF | 0.86 | >12h | — |
| NeuS | 0.84 | >12h | — |
| 3DGS | 1.96 | 11.2m | 113 MB |
| SuGaR | 1.33 | ~1h | 1247 MB |
| **2DGS-15k** | 0.83 | **5.5m** | 52 MB |
| **2DGS-30k** | **0.80** | 10.9m | 52 MB |

SDF 계열 대비 **~100× 스피드업**, SuGaR 대비 3배 이상 빠르고 저장은 1/24. 3DGS의 Chamfer 1.96 → 0.80은 primitive 교체 + 정규화 2개의 효과가 절반 이상이라는 뜻(정규화 ablation: full 0.83 vs w/o normal consistency 1.24, w/o depth distortion 0.88; expected depth 사용 시 0.94, TSDF 대신 SPSR 시 1.07 — Table 5, 15k 기준).

**Tanks and Temples F1 (Table 2)** — 평균: NeuS 0.38, Geo-NeuS 0.35, Neuralangelo 0.50, SuGaR 0.19, 3DGS 0.09, **2DGS 0.32** (시간: implicit >24h vs 2DGS 15.5m). Neuralangelo에는 못 미치지만 explicit 계열에서는 압도적이고 학습은 ~100배 빠르다.

**Mip-NeRF360 PSNR (Table 4, 30k)** — Outdoor: MipNeRF360 24.47 / 3DGS **24.64** / 2DGS 24.34. Indoor: MipNeRF360 **31.72** / 3DGS 30.41 / 2DGS 30.40. NVS는 3DGS 대비 소폭 하락(DTU training-view PSNR도 35.76 vs 34.52) — 기하를 얻는 대가가 화질에 약간 있다.

## 내 실습 연결

리콘랩스에서 mip-splatting → **2DGS** → RaDe-GS를 순서대로 재현·검증하고 프로덕션 채택을 결정한 당사자 입장에서, 이 논문은 "GS는 렌더링용"이라는 통념을 "GS로 계측한다"로 바꾼 전환점이었다. 복원 기반 치수측정 데모(GT 대비 **−1.0/+0.3 mm**)가 성립하려면 뷰마다 흔들리지 않는 depth가 필요한데, 그 요건을 처음 정면으로 푼 것이 2DGS의 multi-view 일관 교차 + depth distortion이다. 특히 normal consistency의 "렌더 노멀 vs 깊이 그라디언트 노멀" 정렬은 TSDF 융합 품질을 좌우하는 핵심이라 실측 파이프라인에서 그대로 체감했다.

다만 검증 과정에서 확인한 트레이드오프도 원문 그대로였다: NVS 화질 소폭 하락, texture-rich 영역 편향 densification(geometry-rich 미세 구조 손해), 정규화 과다 시 over-smoothing. "primitive를 평면으로 바꿔서" 기하를 얻는 2DGS 노선과, "3D 가우시안을 유지한 채 rasterized depth를 닫힌형으로 정확히 계산해서" 기하를 얻는 RaDe-GS 노선의 비교 — 화질 손실 없이 비슷한 Chamfer를 얻을 수 있는가 — 가 최종 채택 판단의 축이었고, 그 이야기는 RaDe-GS 리뷰에서 잇는다.

## 한 줄 평 / 한계

> primitive의 차원을 하나 줄이는 단순한 결정이 multi-view 불일치·노멀 부재·perspective 오차를 한 번에 정리했다 — ray-splat 교차의 두-평면 정식화는 역행렬 없는 닫힌형이라 공학적으로도 아름답다. 다만 원문 스스로 인정하듯 반투명 표면(유리)은 full opacity 가정 탓에 못 다루고, densification이 텍스처 많은 곳을 편애해 미세 기하를 놓치며, 강한 하이라이트 영역에 구멍이 생긴다. "화질 vs 기하" 트레이드오프를 primitive 교체 없이 풀 수 있는가라는 질문이 RaDe-GS로 이어진다.
