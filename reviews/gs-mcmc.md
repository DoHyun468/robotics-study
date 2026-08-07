# 3DGS-MCMC (NeurIPS 2024)

*clone/split/prune 휴리스틱을 전부 버리고 — 가우시안 집합을 "장면 충실도 분포에서 뽑은 MCMC 샘플"로 재해석하면, densification은 확률 보존 재배치가 되고 노이즈 한 항이 초기화 의존을 지운다*

Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi, *3D Gaussian Splatting as Markov Chain Monte Carlo*, NeurIPS 2024. ([arXiv:2404.09591](https://arxiv.org/abs/2404.09591), [프로젝트](https://ubc-vision.github.io/3dgs-mcmc/))

## 한 줄 요약

> 3DGS의 gradient 업데이트에 노이즈 항 하나를 더하면 정확히 SGLD(Stochastic Gradient Langevin Dynamics)가 된다는 관찰에서 출발 — 가우시안들을 $\mathcal{G}\propto\exp(-\mathcal{L}_{\text{total}})$의 MCMC 샘플로 보고, clone/split/prune/opacity-reset 휴리스틱 전부를 **렌더 결과를 보존하는 dead→live 재배치(state transition)** 하나로 대체했다. 가우시안 수는 상한 고정, 초기화는 랜덤이어도 됨: MipNeRF 360 랜덤 초기화에서 3DGS 27.89 → **29.72 PSNR**.

## 문제 — adaptive density control은 휴리스틱 덩어리다

원본 3DGS는 가우시안 배치를 **adaptive density control**로 관리한다: view-space gradient가 크면 clone(작은 가우시안 복제) 또는 split(큰 가우시안 분할), opacity가 낮으면 prune, 그리고 주기적으로 전체 opacity를 리셋해 floater를 제거한다. 저자들이 지적하는 문제:

- **초기화 의존**: 좋은 SfM 포인트 클라우드가 없으면(랜덤 초기화) 품질이 크게 무너진다. densify가 기존 가우시안 근방만 탐색하므로, 초기에 비어 있는 영역은 영영 채워지지 않는다.
- **휴리스틱 튜닝**: gradient 임계값, opacity 리셋 주기 등 다수의 하이퍼파라미터를 장면마다 조심스레 맞춰야 하고, 일부 장면에선 그래도 실패한다.
- **예산 통제 불가**: 최종 가우시안 수가 하이퍼파라미터에서 사전 예측이 안 되어, 메모리/연산 예산을 미리 정해 두고 학습할 수 없다.
- clone 시 렌더 결과가 보존되지 않는다: opacity 0.95짜리를 그대로 복제하면 합성 opacity가 $1-(1-o)^N$으로 커지고 유효 extent도 부풀어(Fig. 1), 학습이 불안정해진다. Bulò et al.의 중심 보정판도 opacity만 고치고 Σ를 안 건드려 불충분.

계보를 하나 짚어두면: 같은 저자들(Kheradmand·Rebain·Yi·Tagliasacchi)의 CVPR 2024 *Soft Mining*이 NeRF 학습 가속에 SGLD를 썼던 팀이다 — 거기선 SGLD로 "어느 픽셀을 학습할지"를 골랐고, 여기선 표현 자체(가우시안 집합)를 분포의 샘플로 본다. 분포 $\mathcal{G}$를 명시적으로 모델링할 필요가 없다는 점이 묘수다: 가우시안들이 이미 그 분포를 표현하고 있다.

## 방법

### 3DGS 업데이트 ≈ SGLD — 노이즈 한 항 차이

3DGS의 파라미터 업데이트(휴리스틱 무시)는

$$\mathbf{g}\leftarrow\mathbf{g}-\lambda_{\text{lr}}\cdot\nabla_{\mathbf{g}}\mathbb{E}_{\mathbf{I}\sim\mathcal{I}}\left[\mathcal{L}_{\text{total}}(\mathbf{g};\mathbf{I})\right]\quad(4)$$

이고, SGLD 업데이트는

$$\mathbf{g}\leftarrow\mathbf{g}+a\cdot\nabla_{\mathbf{g}}\log\mathcal{P}(\mathbf{g})+b\cdot\boldsymbol{\epsilon}\quad(5)$$

이다. 손실을 negative log-likelihood로 두면, 즉

$$\mathcal{G}=\mathcal{P}\propto\exp(-\mathcal{L}_{\text{total}})\quad(6)$$

$\lambda_{\text{lr}}=-a$, $b=0$일 때 두 식은 **동일**하다. 다시 말해 3DGS 학습은 "렌더 품질이 좋을수록 확률이 높은 분포"에서의 노이즈 없는 SGLD였던 것. 노이즈를 되살리면 MCMC 탐색이 된다:

$$\mathbf{g}\leftarrow\mathbf{g}-\lambda_{\text{lr}}\cdot\nabla_{\mathbf{g}}\mathbb{E}_{\mathbf{I}\sim\mathcal{I}}\left[\mathcal{L}_{\text{total}}(\mathbf{g};\mathbf{I})\right]+\lambda_{\text{noise}}\cdot\boldsymbol{\epsilon}\quad(7)$$

(실제 gradient 항은 Adam으로 대체.) 노이즈는 **위치 $\boldsymbol{\mu}$에만** 넣는다 — opacity/scale/color에 넣으면 오히려 성능이 떨어짐을 ablation으로 확인. 그리고 "잘 수렴한" 불투명 가우시안은 흔들지 않도록, opacity 의존 게이트와 가우시안 자신의 이방성 공분산을 씌운다:

$$\boldsymbol{\epsilon}_{\mu}=\lambda_{\text{lr}}\cdot\sigma\!\left(-k(t-o)\right)\cdot\boldsymbol{\Sigma}\,\boldsymbol{\eta},\qquad \boldsymbol{\eta}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),\quad \boldsymbol{\epsilon}=[\boldsymbol{\epsilon}_\mu,\mathbf{0}]\quad(8)$$

$k=100$, $t=0.005$(3DGS의 prune 임계값 중심의 급한 sigmoid 전이). 요컨대 **투명한(dead에 가까운) 가우시안일수록 자기 모양대로 크게 흔들려 탐색하고, 불투명한 가우시안은 gradient가 지배**한다. 노이즈 크기는 학습률과 같이 exponential 스케줄로 감쇠(linear 스케줄은 17.64 PSNR로 붕괴, exponential은 24.21 — TnT).

### 재배치: densify/prune을 확률 보존 state transition으로

휴리스틱의 move/split/clone/prune/add를 전부 하나의 관점으로 통일한다: 가우시안 수가 적은 상태 = "opacity 0짜리 dead 가우시안을 더 가진 동일 상태". 그러면 모든 조작은 샘플 상태 $\mathbf{g}^{old}\to\mathbf{g}^{new}$의 **결정론적 이동**이고, MCMC 체인을 깨지 않으려면 $\mathcal{P}(\mathbf{g}^{new})=\mathcal{P}(\mathbf{g}^{old})$, 즉 **이동 전후 렌더링이 같아야** 한다.

전략: dead($o<0.005$) 가우시안을 live 가우시안 위치로 옮기되, $N-1$개를 $\mathbf{g}_N$에 합류시킬 때 합성 렌더가 보존되도록 파라미터를 재설정한다(부록 A에서 유도):

$$\boldsymbol{\mu}^{new}_{1..N}=\boldsymbol{\mu}^{old}_N,\qquad o^{new}_{1..N}=1-\sqrt[N]{1-o^{old}_N},$$
$$\boldsymbol{\Sigma}^{new}_{1..N}=\left(o^{old}_N\right)^2\left(\sum_{i=1}^{N}\sum_{k=0}^{i-1}\binom{i-1}{k}\frac{(-1)^k\left(o^{new}_N\right)^{k+1}}{\sqrt{k+1}}\right)^{-2}\boldsymbol{\Sigma}^{old}_N\quad(9)$$

opacity 식 $\;(1-o^{new})^N=1-o^{old}$은 Bulò et al.의 $N{=}2$ 보정과 같지만, 핵심은 **Σ까지 줄여야** 한다는 것. α-blending에서 합성값은 여러 가우시안 모양의 곱이라 opacity만 맞추면 유효 extent가 넓어진다(Fig. 1). Σ 재설정 유도는 정확한 MSE 최소화 대신 sliced Wasserstein식 발상 — 중심을 지나는 **임의의 1D 슬라이스에서 적분값 보존** $\int C^{new}dx=\int C^{old}dx$을 요구하면 $(1-p)^N$의 이항 전개를 거쳐 (9)의 폐형 해가 나온다.

구현: 100 iteration마다 실행. 각 dead 가우시안의 목적지는 live들의 **opacity 비례 multinomial 샘플링**으로 선택(모든 이동 결정을 먼저 하고 나서 (9) 적용). Adam moment는 target(원래 있던 가우시안)은 리셋해 안정을 주고, 이주해 온 새 가우시안은 유지해 탐색을 살린다.

### 예산 고정 + L1 정규화로 "낭비되는 가우시안" 회수

가우시안 수는 **상한 고정**: 처음 정한 수로 시작하고(랜덤 초기화 시 카메라 bounding box 3배 내 균일 100k), live 수를 상한까지 5%씩 늘린다. 불필요한 가우시안이 스스로 죽어(재배치 대상이 되어) 유용한 곳으로 respawn하도록 opacity와 scale에 L1을 건다:

$$\mathcal{L}_{\text{total}}=(1-\lambda_{\text{D-SSIM}})\mathcal{L}_1+\lambda_{\text{D-SSIM}}\mathcal{L}_{\text{D-SSIM}}+\lambda_o\sum_i|o_i|_1+\lambda_{\boldsymbol{\Sigma}}\sum_{ij}\left|\sqrt{\text{eig}_j(\boldsymbol{\Sigma}_i)}\right|_1\quad(10)$$

$\lambda_{\text{noise}}=5\times10^5$, $\lambda_{\boldsymbol{\Sigma}}=0.01$, $\lambda_o=0.01$(Deep Blending만 0.001). warmup 500 iter 동안은 재배치·증식 없음.

**원본 3DGS 대비 바뀐 것 요약**:

1. 위치 업데이트에 opacity-게이트 이방성 노이즈 (8) 추가 — Adam 위에 얹음, 스케줄은 학습률과 동일한 exponential 감쇠($1.6e{-4}\to1.6e{-6}$).
2. adaptive density control + opacity reset 전부 제거 → 100-iter 주기 재배치 (9)로 대체.
3. 손실에 opacity·scale(공분산 고유값 제곱근) L1 추가.
4. 가우시안 수 상한 고정, live 수 5%씩 점증. 렌더러와 표현 자체는 그대로라 추론 속도는 3DGS와 동일.

## 결과 (원문 수치)

**같은 가우시안 수 예산** (Table 1, PSNR/SSIM/LPIPS, 3회 평균):

| 데이터셋 | 3DGS (Random) | Ours (Random) | 3DGS (SfM) | Ours (SfM) |
|---|---|---|---|---|
| NeRF Synthetic | 33.42/0.97/0.04 | **33.80**/0.97/0.04 | – | – |
| MipNeRF 360 | 27.89/0.84/0.26 | **29.72**/0.89/0.19 | 29.30/0.88/0.21 | **29.89**/0.90/0.19 |
| Tanks & Temples | 21.93/0.79/0.27 | **24.21**/0.86/0.19 | 23.67/0.84/0.22 | **24.29**/0.86/0.19 |
| Deep Blending | 29.55/0.90/0.33 | **29.71**/0.90/0.32 | 29.64/0.90/0.32 | 29.67/0.89/0.32 |
| OMMO | 28.24/0.88/0.24 | **29.31**/0.90/0.20 | 28.83/0.89/0.22 | **29.52**/0.91/0.20 |

- **랜덤 초기화 강건성이 핵심**: 3DGS는 Random↔SfM 격차가 크지만(MipNeRF 360에서 27.89 vs 29.30, TnT 21.93 vs 23.67), 본 방법은 Random 29.72 vs SfM 29.89로 거의 무차이 — 초기화와 무관하게 3DGS를 이긴다. 저자들은 MipNeRF 360에서 NeRF 백본(MipNeRF360 29.23)을 3DGS 계열로 처음 넘었다고 주장.
- **초기화 범위 ablation** (Table 2, MipNeRF 360 랜덤): 카메라 extent 1×로 좁혀 초기화하면 3DGS는 22.72/0.75/0.34로 붕괴, 본 방법은 29.64/0.89/0.19로 유지.
- **제한 예산** (Fig. 3): 100k~2M 전 구간에서 우위, 예산이 작을수록 격차 확대.
- **컴포넌트 ablation** (Table 3, MipNeRF 360 랜덤): 전체 29.72 vs 노이즈 제거($\lambda_{\text{noise}}{=}0$) 27.41 vs 모든 파라미터에 노이즈 29.11 vs 정규화만 3DGS에 이식 23.84(오히려 해로움). 노이즈 설계에서 공분산 항 제거 시 TnT 24.21→23.16, opacity 게이트 제거 시 22.47.
- **시간/메모리** (Table 4, Room·1.5M): iteration당 80ms vs 3DGS 76ms(재배치의 multinomial 샘플링이 naive PyTorch 구현임에도 오버헤드 미미). 3DGS 31.7 PSNR/25분 대비 $\lambda_o{=}0.001$ 설정 32.4/30분, 300k 예산이면 31.8/21분 — 비슷한 PSNR 기준으론 더 빠름. gsplat의 독립 재구현에서 학습 시간 20%·메모리 65% 절감 보고.
- 장면별로 보면(Table 5) 격차가 특히 큰 곳이 랜덤 초기화의 대형 야외 장면: OMMO #10 29.64→31.20, TnT Truck 22.63→26.02, MipNeRF 360 Stump 23.91→27.67. 노이즈 항을 끄면 정확히 이런 장면에서 시야 밖/미탐색 영역이 비는 것을 Fig. 4가 시각적으로 보여준다.

## 내 실습 연결

리콘랩스 3DGS 프로덕션에서 내가 잡던 문제 축 — floater 억제·배경 모델링, mip 이슈 재현 끝의 RaDe-GS 채택 — 은 결국 전부 densify 휴리스틱과 opacity reset의 부작용을 사후에 수습하는 일이었다. MCMC 관점이 실무에 주는 함의는 두 가지가 선명하다. 첫째, **예산 고정 운영**: 상품 3D 뷰어는 기기별 메모리/프레임 예산이 정해져 있는데, 3DGS는 최종 가우시안 수를 통제 못해 장면마다 사후 prune 튜닝이 필요했다. 이 프레임에선 상한을 선언하고 시작하며 L1이 낭비분을 스스로 회수한다(gsplat 재구현의 메모리 65% 절감이 그 증거). 둘째, **저텍스처 캡처 강건성**: 흰 배경·매끈한 제품처럼 COLMAP 포인트가 부실한 캡처에서 3DGS 품질이 흔들리는 걸 겪었는데, Table 2의 1× extent 실험(22.72 vs 29.64)이 정확히 "초기 포인트가 장면을 못 덮는" 상황의 해법을 보여준다. floater 관점에서도 opacity reset이라는 전역 충격요법 대신 dead 가우시안의 국소 재배치로 대체되는 구조가 훨씬 예측 가능하다. gsplat에 이미 들어가 있으니 RaDe-GS류 depth 정칙화와의 조합 실험이 다음 스텝으로 자연스럽다.

## 한 줄 평 / 한계

**한 줄 평**: "휴리스틱을 지우는 논문"은 많지만, 지운 자리에 SGLD 동형성이라는 원리와 렌더 보존 폐형 해 (9)를 놓았다는 점에서 드물게 깔끔하다 — 특히 clone이 왜 학습을 불안정하게 했는지(extent 팽창, Fig. 1)를 처음 제대로 설명했다.

한계는 저자 스스로 명시한 것 포함: ① 표현 자체는 3DGS 그대로라 aliasing(Mip-Splatting류)·반사 모델링 한계를 공유 — "더 나은 학습 프레임"이지 표현 개선이 아니다. ② $\mathcal{P}(\mathbf{g}^{new})\approx\mathcal{P}(\mathbf{g}^{old})$는 근사라 100-iter 간격이라는 새 주기 하이퍼가 남고, $\lambda_{\text{noise}}$·$\lambda_o$·$\lambda_\Sigma$·noise 스케줄 등 "휴리스틱 제거" 뒤에도 튜닝 손잡이가 아주 없진 않다(스케줄 선택에 PSNR 17.64~24.21 낙차). ③ 최고 PSNR까지 가려면 3DGS보다 오래 걸린다(25분→42분, $\lambda_o{=}0.01$). ④ MCMC라 부르지만 detailed balance를 만족하는 정통 샘플러는 아니고(결정론적 이동 + SGLD), 후험 분포 탐색이 아닌 최적화 관점의 재해석에 가깝다.
