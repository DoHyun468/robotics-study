# RoMa (CVPR 2024)

*"coarse는 견고하게, fine은 정밀하게" — DKM의 뼈대 위에 frozen DINOv2 + 전용 fine encoder의 분업, anchor 분류형 Transformer 디코더, regression-by-classification→robust regression 손실을 얹어, 극한 조건 매칭(WxBS)을 36% 끌어올린 DKM의 정식 후속작*

Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, Michael Felsberg, *RoMa: Robust Dense Feature Matching*, CVPR 2024. Linköping University · ECUST · Chalmers.
([arXiv:2305.15404](https://arxiv.org/abs/2305.15404) · [코드](https://github.com/Parskatt/RoMa))

## 한 줄 요약

> DKM 파이프라인에서 (1) coarse feature를 **frozen DINOv2**로, fine feature를 **전용 VGG19**로 분리하고, (2) coarse 매칭을 좌표 회귀가 아닌 **K=64×64 anchor 확률 분류**를 내놓는 Transformer match decoder로 바꾸고, (3) 손실을 coarse **regression-by-classification** + fine **generalized Charbonnier robust regression**으로 재설계해 — MegaDepth AUC@5° 60.4→**62.6**, ScanNet 29.4→**31.8**, 그리고 극한 벤치 WxBS에서 58.9→**80.1**(+36%)을 달성한 robust dense matcher.

## 문제

dense matching의 coarse-to-fine 파이프라인은 종래 3D 지도학습으로 coarse feature까지 통째로 학습했다. 원문이 짚는 문제는 두 겹이다.

- **Robustness**: 실측 3D 데이터셋은 규모가 제한적이라 coarse feature가 학습 장면에 과적합한다 — 학습 분포에서 벗어난 극한 조건(조명·계절·스케일·시점 변화, WxBS류)에서 무너진다. backbone을 얼리는 것이 과적합 방지의 정석이지만, ImageNet 분류 사전학습 frozen backbone은 매칭에 부적합하다(Tab. 1). MIM 계열 self-supervised 모델(DINOv2)은 강건한 all-purpose feature를 주지만 **stride 14짜리 coarse feature뿐**이라 정밀 localization용 fine feature가 없다.
- **정식화**: DKM은 coarse에서도 fine에서도 non-robust 회귀 손실을 쓴다. 그러나 coarse scale의 매치 분포는 motion boundary 근처에서 **다봉(multimodal)** — 회귀는 봉우리 사이 평균으로 붕괴한다. 반대로 이전 warp에 조건화된 refinement 분포는 국소적으로 unimodal이라 다른 처방이 필요하다.

RoMa의 답: coarse와 fine을 feature·모듈·손실 세 층위 모두에서 **다른 문제로 분리해 각각 맞는 도구를 쓴다**.

## 방법

표기는 DKM을 따른다. dense warp $W^{\mathcal{A}\to\mathcal{B}}$와 matchability $p(x^{\mathcal{A}})$를 추정하며, coarse global matcher $G_\theta = D_\theta \circ E_\theta$(match encoder + decoder) 후 refiner $R_\theta$가 stride $\{1,2,4,8\}$에서 residual을 재귀 예측한다(식 (2)–(4)). **GP match encoder $E_\theta$(지수 코사인 커널, inverse temperature 10)와 refiner 구조·gradient detach는 DKM 그대로 유지**하고, 나머지를 교체한다.

### 1. Robust + Localizable Features — frozen DINOv2 coarse / 전용 ConvNet fine (§3.2)

먼저 frozen feature의 매칭 적성을 분리 측정한다: frozen backbone 위에 linear layer 하나 + kernel NN matcher만 학습, MegaDepth 448×448에서 EPE와 Robustness%(오차 32px 미만 매치 비율)를 잰다(Tab. 1).

| Backbone | EPE ↓ | Robustness % ↑ |
|---|---|---|
| VGG19 | 87.6 | 43.2 |
| RN50 | 60.2 | 57.5 |
| **DINOv2** | **27.1** | **85.6** |

DINOv2(ViT-L/14, patch feature만 사용, 1024→512 linear 투영)가 시점 변화에 압도적으로 강건하다. 그래서 $F_{\text{coarse},\theta}=$ **frozen DINOv2**로 두는데 — 얼려두는 것 자체가 (a) 학습셋 과적합을 막는 robustness prior이고 (b) 계산·메모리도 절약이다. 단 DINOv2는 fine feature가 없으므로 $F_{\text{fine},\theta}$를 따로 둔다. 여기서 두 가지 관찰:

- coarse/fine 인코더의 **가중치 공유를 끊는 것만으로** 성능이 오른다(Tab. 2 Setup II) — 각자 자기 과업에 특화(specialization)되기 때문.
- fine encoder로는 **VGG19가 RN50을 능가**한다(Setup III). coarse 매칭에서는 VGG19가 RN50보다 나빴다는 Tab. 1과 뒤집힌 결과 — **fine localizability와 coarse robustness 사이에 내재적 긴장**이 있다는 증거이고, 분업 설계의 정당화다. fine feature는 stride $\{1,2,4,8\}$, 차원 $\{64,128,256,512\}$를 $\{9,64,256,512\}$로 투영해 쓴다.

### 2. Transformer Match Decoder — 회귀 대신 anchor 분류 (§3.3)

DKM의 ConvNet 디코더는 warp 좌표를 직접 회귀했다. RoMa는 출력 공간을 이산화한 **regression-by-classification**으로 바꾼다:

$$p_{\text{coarse},\theta}(x^{\mathcal{B}}|x^{\mathcal{A}}) = \sum_{k=1}^{K}\pi_k(x^{\mathcal{A}})\,\mathcal{B}_{m_k}$$

$K=64\times64$개 anchor를 이미지 grid의 빈틈없는 타일로 깔고, $\mathcal{B}=\mathcal{U}$(각 셀 위 균일분포), $\pi_k$가 anchor 확률이다. 회귀보다 강건한 이유는 **다봉성 표현력**: 반복 구조·motion boundary에서 조건분포가 여러 봉우리를 가질 때 분류는 그대로 담지만 회귀는 봉우리 평균이라는 존재하지 않는 위치로 붕괴한다. refinement에 넘길 때는 argmax anchor $k^*(x)=\arg\max_k \pi_k(x)$ 뒤 4-이웃 국소 softargmax로 결정적 warp를 만든다:

$$\hat{W}^{\mathcal{A}\to\mathcal{B}}_{\text{coarse}} = \frac{\sum_{i\in N_4(k^*)}\pi_i m_i}{\sum_{i\in N_4(k^*)}\pi_i}$$

디코더 구조도 교체 근거가 명확하다. ConvNet coarse 디코더는 학습 해상도에 과적합하고 locality에 과의존해 coarse warp를 oversmoothing한다. 그래서 **position encoding 없는 ViT 블록 5개**(8 heads, hidden 1024, MLP 4096)로, feature 유사도로만 전파하게 제한하니 훨씬 강건해졌다. 입력은 투영된 DINOv2 512차원 + GP 출력 512차원 concat, 출력은 $B\times H\times W\times(K{+}1)$ (anchor 확률 $K$ + matchability 1). 이 디코더는 최종 손실과 결합할 때 특히 유효하며, 완성본에서 ConvNet 디코더로 되돌리면 성능이 뚜렷이 떨어진다(Setup VIII).

### 3. Robust Loss Formulation — 이론 모델에서 손실 유도 (§3.4)

scale $s$에서의 matchability를 무한 해상도의 정확한 매핑에 Gaussian을 컨볼브한 **스케일 확산**으로 모델링한다:

$$q(x^{\mathcal{A}}, x^{\mathcal{B}}; s) = \mathcal{N}(0, s^2\mathbf{I}) * p(x^{\mathcal{A}}, x^{\mathcal{B}}; 0)$$

motion boundary 불연속 근처에서 이 확산이 조건분포를 다봉으로 만든다(Fig. 3) — coarse엔 다봉 표현이 필수인 이유. 반면 좋은 초기 warp에 조건화된 refinement는 국소 unimodal이되, **초기값이 분포 support 밖이면 non-robust 손실이 문제**가 되므로 robust 회귀가 필요하다. 손실은 각 scale에서 이론 분포 $q$와 모델 분포 사이 **KL divergence 최소화**로 유도한다.

**Coarse**: 비겹침 anchor bin에서 KL을 전개하면(식 (11)–(13)) $-\int \log\pi_{k^\dagger}(x^{\mathcal{A}}) + \lambda\log p_{\text{coarse},\theta}(x^{\mathcal{A}})\,dq$ — $k^\dagger(x)=\arg\min_k\|m_k-x\|$(GT에 가장 가까운 anchor)에 대한 **cross-entropy 분류 손실**이 된다. matchability 항은 DKM처럼 $\lambda$ 가중 binary cross-entropy. 이것이 $\mathcal{L}_{\text{coarse}}$.

**Fine**: refinement 출력을 **generalized Charbonnier**($\alpha=0.5$) 분포로 모델링하고 refiner가 평균 $\mu$를 추정한다. 로그밀도는 (상수·스케일 무시하면)

$$\log p_\theta(x^{\mathcal{B}}_i|x^{\mathcal{A}}_i,\hat{W}^{\mathcal{A}\to\mathcal{B}}_{i+1}) = -\big(\|\mu_\theta(x^{\mathcal{A}}_i,\hat{W}_{i+1})-x^{\mathcal{B}}_i\|^2 + s\big)^{1/4}$$

$s=2^i c$, $c=0.03$. gradient가 국소에서는 L2처럼 행동하고 멀리서는 $|x|^{-1/2}$로 감쇠(Fig. 4) — DKM의 clipped L2를 대체하는 robust regression이다. fine scale $i\in\{0,1,2,3\}$마다 KL(식 (17)–(18))을 두고, matchability는 BCE, 전 scale 합산이 $\mathcal{L}_{\text{fine}}$. 최종 $\mathcal{L}=\mathcal{L}_{\text{coarse}}+\mathcal{L}_{\text{fine}}$이며, matching과 refinement 사이 gradient가 끊겨 있고 인코더도 비공유라 **두 손실 간 가중 튜닝이 불필요**하다.

### 학습 세팅 (§4.2)

DKM과 동일 세팅: lr $10^{-4}$(디코더)·$5\cdot10^{-6}$(인코더) at batchsize 8, MegaDepth+ScanNet 학습 split(테스트 장면 제외), MVS depth(MegaDepth)·RGB-D(ScanNet) warp 지도. ScanNet-1500 평가만 ScanNet 학습 모델, 나머지는 전부 MegaDepth 단독 학습 모델. ablation 448×448, 최종 모델 560×560.

## 결과 (원문 수치)

**Ablation (Tab. 2, 100-PCK ↓ @1px/3px/5px)** — 기여가 단계별로 쌓인다:

| Setup | 1px | 3px | 5px |
|---|---|---|---|
| I DKM baseline (재학습) | 17.0 | 7.3 | 5.8 |
| II coarse/fine 인코더 분리(RN50/RN50) | 16.0 | 6.1 | 4.5 |
| III fine=VGG19 | 14.5 | 5.4 | 4.5 |
| IV + Transformer decoder(회귀) | 14.4 | 5.4 | 4.1 |
| V + coarse=frozen DINOv2 | 14.3 | 4.6 | 3.2 |
| VI + reg-by-classification | 13.6 | 4.1 | 2.8 |
| **VII + robust fine loss (RoMa)** | **13.1** | **4.0** | **2.7** |
| VIII − Transformer→ConvNet decoder | 14.0 | 4.9 | 3.5 |

**벤치마크** (모두 원문 Tab. 3–8):

- **WxBS** (extreme viewpoint+illumination, mAA@10px): LoFTR 55.4, DKM 58.9 → **RoMa 80.1**. 원문이 "outstanding improvement of **36%**"로 명시한 핵심 결과 — 강건성 설계가 정확히 겨냥한 지점.
- **IMC2022** (Google street-view fundamental matrix, mAA@10): DKM 83.1, ASpanFormer 83.8 → **RoMa 88.0** (상대 오차 26% 감소).
- **MegaDepth-1500** (AUC@5°/10°/20°): DKM 60.4/74.9/85.1 → **RoMa 62.6/76.7/86.3**.
- **ScanNet-1500**: DKM 29.4/50.7/68.3 → **RoMa 31.8/53.4/70.9** — AUC@20° 70 첫 돌파.
- **MegaDepth-8-Scenes**: DKM 60.5/74.5/84.2 → **RoMa 62.2/75.9/85.3**.
- **InLoc 시각 측위** (HLoc, 0.25/0.5/1.0m): DUC1 **60.6/79.3/89.9**, DUC2 **66.4/83.2/87.8** — DKM(51.5/76.8/86.9, 63.4/82.4/87.8) 대비 전반 우위.
- **런타임**: 560×560, batch 8, RTX6000에서 DKM 186.3 → RoMa **198.8 ms/pair** — 7% 증가로 강건성을 산 셈.

## 내 실습 연결

석사에서 선체블록 wide-baseline 스테레오 대응에 SuperGlue·DKM·LoFTR 앙상블을 실사용했다(대응 오차 29.1→2.23px). **지금 같은 문제를 다시 푼다면 RoMa가 1순위 후보다.** 이유는 벤치 수치가 아니라 실패 축의 일치다.

- 당시 앙상블을 짠 근본 이유가 저텍스처 강판 + wide-baseline에서 **단일 매처의 coarse 매칭이 무너지는** 실패 모드였는데, 그게 정확히 RoMa가 WxBS +36%로 증명한 robustness 축이다. DKM이 무너지던 pair에서 RoMa가 살아남는 원문 Fig. 6의 양상은, 앙상블로 메우던 구멍을 매처 하나가 흡수할 수 있다는 뜻 — 앙상블 자체를 줄이거나 없앨 후보다.
- **기대**: frozen DINOv2 prior는 MegaDepth(관광지 건물)에 과적합하지 않은 일반 feature이므로, 학습 분포에 없는 산업 표면으로의 도메인 갭에서 fine-tuned coarse feature보다 덜 무너질 것이라 기대할 수 있다 — Tab. 1의 Robustness 85.6%가 그 방향의 증거다. 반복 용접선·대칭 구조에서 다봉성을 유지하는 anchor 분류도 강판의 반복 패턴과 궁합이 좋을 것.
- **우려**: DINOv2 사전학습 분포(웹 자연 이미지)에 무텍스처 금속 표면·정반사가 얼마나 있었는지는 미지수다. semantic 단서가 거의 없는 균질 강판에서 DINOv2 coarse feature가 무엇을 붙잡을지는 실측 검증 사안이고, certainty 임계·balanced sampling 파이프라인(DKM 유산이라 그대로 이식 가능)으로 걸러보며 확인해야 한다.

## 한 줄 평 / 한계

> "무엇을 학습하고 무엇을 얼릴 것인가, 무엇이 다봉이고 무엇이 단봉인가"를 coarse/fine 축으로 정확히 갈라 각각 맞는 도구를 배정한 논문 — 새 모듈 발명보다 **분업의 정식화**가 승부처였고, 그 결과 dense matcher가 처음으로 극한 조건에서 신뢰할 만한 도구가 됐다.

한계는 원문 명시 두 가지 + 읽는 쪽 관찰. (1) **supervised 대응에 의존** — 쓸 수 있는 데이터 양이 제한되며, frozen foundation feature로 완화했을 뿐 해소는 아니다. (2) dense matching은 downstream(two-view geometry·측위·복원)의 **간접 최적화** — downstream 직접 학습이 남은 방향. (3) ablation을 뜯으면 Transformer 디코더는 회귀 상태(Setup IV)에서는 이득이 미미하고 reg-by-classification 손실과 결합해야(VI) 효과가 서는 등 기여들이 상호의존적이다 — 개별 요소의 독립 이식은 조심할 것. (4) ViT-L/14 frozen이라도 DINOv2를 얹은 만큼 메모리 발자국은 커지며, 런타임 +7%는 RTX6000 batch 8 기준이라 엣지 배치에서는 별도 프로파일이 필요하다.
