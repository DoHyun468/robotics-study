# LightGlue (ICCV 2023)

*SuperGlue를 해부해 병목(Sinkhorn, 고정 깊이)을 걷어내고, "쉬운 쌍은 빨리, 어려운 쌍은 깊게"라는 적응형 추론을 얹은 재설계.*

Philipp Lindenberger, Paul-Edouard Sarlin, Marc Pollefeys, *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023. ([arXiv](https://arxiv.org/abs/2306.13643), [코드](https://github.com/cvg/LightGlue))

## 한 줄 요약
> SuperGlue의 설계 결정들을 하나씩 재검토해 — optimal transport(Sinkhorn) 제거, similarity와 matchability의 분리 예측, rotary 상대 위치 인코딩, deep supervision — 더 정확하고 훈련이 쉬운 매처를 만들고, 여기에 이미지 쌍의 난이도에 따라 **깊이(조기 종료)와 폭(포인트 pruning)을 스스로 조절하는 적응형 추론**을 더해 SuperGlue 대비 최대 2.5–4배 빠른 drop-in 대체품을 완성했다.

## 문제 — SuperGlue의 병목 진단

SuperGlue는 sparse matching의 표준이 됐지만, 저자들은 세 가지 구조적 비용을 짚는다.

- **Sinkhorn의 비용**: 미분 가능한 optimal transport는 row/column 정규화를 수십 회(100 iteration) 반복해야 해 연산·메모리 모두 비싸다. dustbin(무매칭 버킷)이 모든 점의 similarity score를 한 행렬에 얽어 학습 dynamics도 나쁘게 만든다.
- **고정 깊이 연산**: 시각적 중첩이 크고 외형 변화가 적은 "쉬운" 쌍도, wide-baseline의 "어려운" 쌍과 똑같이 전체 레이어를 통과한다. Sinkhorn이 비싸서 중간 레이어에서 예측을 뽑을 수 없고, 마지막 레이어에서만 supervision을 받는다.
- **훈련 난이도**: 절대 위치를 MLP로 초기에 융합하면 깊은 레이어에서 위치 정보가 잊힌다. 결과적으로 원조 SuperGlue의 성능을 후속 재현들이 따라잡지 못할 만큼 훈련이 어렵다.

## 방법 — SuperGlue 대비 무엇을 바꿨나

설정은 SuperGlue와 동일: 이미지 $A, B$에서 정규화된 위치 $\mathbf{p}_i$와 descriptor $\mathbf{d}_i \in \mathbb{R}^d$를 가진 $M, N$개 local feature로부터 soft partial assignment $\mathbf{P} \in [0,1]^{M \times N}$을 예측한다. 백본은 $L{=}9$개의 동일한 레이어(각각 self-attention + cross-attention 유닛, 4 heads, $d{=}256$).

### 1. Transformer 백본 — rotary 상대 위치 인코딩

각 유닛에서 상태는 상대 이미지 $S$로부터 모은 메시지로 residual 업데이트된다 (Eq. 1–2):

$$
\mathbf{x}_i^I \leftarrow \mathbf{x}_i^I + \mathrm{MLP}\left(\left[\mathbf{x}_i^I \,|\, \mathbf{m}_i^{I \leftarrow S}\right]\right), \quad
\mathbf{m}_i^{I \leftarrow S} = \sum_{j \in \mathcal{S}} \mathrm{Softmax}\big(a_{ik}^{IS}\big)_j \, \mathbf{W} \mathbf{x}_j^S .
$$

**Self-attention**은 절대 위치 대신 rotary encoding으로 상대 위치를 본다 (Eq. 3):

$$
a_{ij} = \mathbf{q}_i^\top \, \mathbf{R}(\mathbf{p}_j - \mathbf{p}_i) \, \mathbf{k}_j ,
$$

$\mathbf{R}(\cdot) \in \mathbb{R}^{d \times d}$는 공간을 $d/2$개의 2D subspace로 나눠 학습된 basis $\mathbf{b}_k \in \mathbb{R}^2$에 투영한 각도만큼 회전시키는 block-diagonal 행렬이다(Eq. 4). 핵심 관찰: 사영 기하에서 시각적 관측의 위치는 카메라의 이미지 평면 내 translation에 대해 **상대 거리만 불변**이므로, 절대 위치가 아닌 상대 위치를 인코딩해야 한다. 인코딩은 value에는 적용하지 않아 상태에 스며들지 않고, 모든 레이어에서 동일해 한 번만 계산해 캐싱한다.

**Cross-attention**은 key만 계산하는 bidirectional attention으로 similarity를 한 번만 계산한다 (Eq. 5):

$$
a_{ij}^{IS} = \mathbf{k}_i^{I\top} \mathbf{k}_j^S \stackrel{!}{=} a_{ji}^{SI} ,
$$

$O(NMd)$인 이 단계에서 정확도 손실 없이 절반의 비용을 아낀다(ablation: 22.8ms → 19.4ms). 상대 위치는 이미지 간에는 의미가 없으므로 cross에는 위치 인코딩을 넣지 않는다.

### 2. Assignment 재설계 — Sinkhorn 제거

similarity와 matchability를 **분리해** 예측한 뒤 곱으로 결합한다 (Eq. 6–8):

$$
\mathbf{S}_{ij} = \mathrm{Linear}(\mathbf{x}_i^A)^\top \mathrm{Linear}(\mathbf{x}_j^B), \qquad
\sigma_i = \mathrm{Sigmoid}\big(\mathrm{Linear}(\mathbf{x}_i)\big) \in [0,1],
$$

$$
\mathbf{P}_{ij} = \sigma_i^A \, \sigma_j^B \, \underset{k \in \mathcal{A}}{\mathrm{Softmax}}(\mathbf{S}_{kj})_i \, \underset{k \in \mathcal{B}}{\mathrm{Softmax}}(\mathbf{S}_{ik})_j .
$$

$\sigma_i$는 점 $i$가 상대 이미지에 대응점을 가질 가능성(occlusion이면 $\sigma_i \to 0$)이고, 두 방향 softmax의 곱이 mutual nearest neighbor를 soft하게 구현한다. 저자들 표현으로는 "mutual NN 탐색과 학습된 inlier 분류기의 우아한 융합"이며 Sinkhorn보다 훨씬 빠르고, dustbin이 만들던 score 얽힘이 사라져 gradient도 깨끗하다. 최종 correspondence는 양쪽 다 matchable하고 $\mathbf{P}_{ij}$가 행·열 최대이면서 $\tau{=}0.1$을 넘는 쌍.

### 3. 적응형 추론 — depth와 width

**① 조기 종료(depth-adaptive)**: 각 레이어 끝에서 compact MLP가 각 점의 예측이 "최종 레이어와 같을지"의 confidence를 출력한다 (Eq. 9):

$$
c_i = \mathrm{Sigmoid}\big(\mathrm{MLP}(\mathbf{x}_i)\big) \in [0,1] ,
$$

전체 점 중 confident한 비율이 $\alpha$를 넘으면 추론을 멈춘다 (Eq. 10):

$$
\text{exit} = \left( \frac{1}{N+M} \sum_{I \in \{A,B\}} \sum_{i \in \mathcal{I}} [\![ c_i^I > \lambda_\ell ]\!] \right) > \alpha .
$$

분류기 자체가 초기 레이어에서 덜 정확하므로 임계값을 레이어에 따라 지수 감쇠시킨다: $\lambda_\ell = 0.8 + 0.1 e^{-4\ell/L}$ (Eq. 12). MLP의 오버헤드는 최악의 경우에도 추론 시간의 2%.

**② 포인트 pruning(width-adaptive)**: confident한데 matchability가 낮은 점 — 비가시 영역·비반복 keypoint — 은 이후 레이어에 도움이 안 되므로 매 레이어에서 제거한다 (Eq. 13):

$$
\text{unmatchable}(i) = c_i^\ell > \lambda_\ell \;\, \& \;\, \sigma_i^\ell < \beta, \qquad \beta = 0.01 .
$$

attention이 점 수에 이차이므로 pruning 효과는 점이 많을수록 커진다. 몇 레이어만에 30% 이상의 keypoint가 제외된다(Fig. 11).

### 4. 학습 레시피

- **Deep supervision**: 가벼운 head 덕에 모든 레이어 $\ell$에서 assignment를 예측·supervise할 수 있다(Eq. 11: 각 레이어의 $\log {}^\ell\mathbf{P}_{ij}$ + unmatchable 점의 $\log(1-{}^\ell\sigma)$). 이것이 조기 종료를 가능케 하는 전제이자 수렴 가속의 열쇠다.
- **2단계 학습**: 먼저 correspondence를 학습하고, 그 다음 confidence 분류기를 학습한다(각 레이어의 예측이 최종 레이어와 같은지의 binary cross-entropy). 분류기 gradient는 상태로 전파하지 않아 매칭 정확도에 영향이 없다.
- **사전학습 + 파인튜닝**: Oxford-Paris 1M distractor에서 뽑은 170k 이미지에 극단적 perspective/photometric augmentation을 건 합성 homography로 사전학습(40 epochs, 6M pairs, RTX 3090 2장으로 2일)한 뒤 MegaDepth(196개 랜드마크, 368/5/24 scene split)로 파인튜닝. 사전학습 생략이 후속 연구들의 실패 원인이라고 지적한다.
- **세부**: 이미지당 keypoint 1k가 아닌 2k, gradient checkpointing + mixed precision으로 24GB GPU 한 장에 batch 32. epipolar error가 큰 점을 unmatchable로 라벨링. Fig. 5: 5M pairs(2 GPU-days)만에 SuperGlue 대비 −33% loss, +4% recall — SuperGlue는 비슷한 정확도에 7일 이상 걸린다.

## 결과 — 원문 수치

- **합성 homography ablation (Table 4)**: SuperGlue precision 74.6 / recall 90.5 / 29.1ms → LightGlue **86.8 / 96.3 / 19.4ms** (+12% P, +4% R). matchability 제거 시 precision 67.4로 붕괴, 절대 위치로 바꾸면 84.2.
- **HPatches homography (Table 1)**: precision 88.9로 sparse 매처 최고(SuperGlue 87.4), DLT AUC@1px 35.9로 단순 solver를 MAGSAC급으로 끌어올림. @5px 78.6은 dense LoFTR(70.6)보다 높다.
- **MegaDepth-1500 pose (Table 2, SuperPoint)**: RANSAC AUC@5/10/20° = 49.9/67.0/80.1, 44.2ms — SuperGlue(49.7/67.1/80.6, 70.0ms) 대비 동급 정확도에 30% 빠름. **adaptive**는 49.4/67.2/80.1을 31.4ms에 내 SuperGlue보다 2배 이상 빠르다. LO-RANSAC에서는 66.7/79.3/87.9로 LoFTR(66.4/78.6/86.5, 181ms)·MatchFormer(388ms)를 5–11배 빠른 속도로 능가. ASpanFormer(69.4/81.1/88.9, 369ms)만이 더 정확하다.
- **적응형 트레이드오프 (Table 5, 11)**: 평균 5.7층에서 정지(easy 4.7 / hard 6.9), 점 23.7% pruning, 전체 33% 시간 절감. exit $\alpha{=}95\%$면 시간 70.6%에 AUC@5° 66.7→66.3, $\alpha{=}80\%$면 시간 48.4%에 65.2 — 정확도 손실 대비 절감 폭이 층수 고정 트리밍(layer 5/9: 60% 시간에 65.0)보다 유리하다.
- **Aachen Day-Night (Table 3)**: 88.2/95.5/98.7(day), 86.7/92.9/100(night)의 SuperGlue와 사실상 동급인 89.2/95.4/98.5, 87.8/93.9/100을 내면서 처리량 6.5 → 17.2 pairs/s(**2.5배**), efficient self-attention을 켠 optimized는 26.1 pairs/s(**4배**). 4096 keypoints를 실시간 매칭.
- **IMC 2020 SfM (Table 6)**: SP+LightGlue가 multiview AUC@5/10/25° 62.87/79.36/86.98로 SuperGlue(61.88/78.97/86.75)를 앞서며 16.2 → 43.4 pairs/s. DISK+LightGlue는 stereo 67.02/77.82로 큰 격차의 최고. IMC 2023 public/private에서 SP+SuperGlue 36.1/43.8 → SP+LightGlue 38.4/46.1(+2.3%).

## 내 실습 연결

석사 계측 파이프라인에서 SuperGlue를 앙상블의 일원으로 실사용했다(멀티뷰 매칭 → 29.1→2.23px 오차 파이프라인). 그 경험에서 "오늘 다시 고른다면?"에 대한 이 논문의 답은 명확하다 — **drop-in으로 LightGlue**. 근거는 세 겹이다. ① 정확도가 같거나 높으면서(MegaDepth 동급, homography는 precision +1.5, IMC는 우세) 기본 30–35%, adaptive로 2배 이상 빠르다. 내 파이프라인은 오프라인이라 latency는 안 급하지만 멀티뷰 전체 쌍을 도는 **처리량**이 곧 실험 회전 속도였다. ② 앙상블 관점에서 흥미로운 건 depth-adaptive의 철학이다 — 내 앙상블도 사실 "쉬운 쌍은 아무 매처나 맞고, 어려운 쌍이 문제"였는데, LightGlue는 그 난이도 분기를 모델 내부의 confidence 분류기로 내재화했다. 쌍별 매처 선택 휴리스틱을 짜는 대신 $\alpha$ 하나로 정확도-속도를 조절할 수 있다. ③ matchability $\sigma_i$가 별도 score로 나오는 것도 계측에 실용적이다 — Sinkhorn dustbin에 얽힌 값이 아니라 점 단위 독립 신호라, 반사·가림 영역의 keypoint를 downstream 필터링에 바로 쓸 수 있다. 단, 내 도메인(산업 계측 대상)은 MegaDepth의 관광지 분포와 멀어서 합성 homography 사전학습 → 도메인 파인튜닝 레시피를 그대로 따라야 할 텐데, 24GB GPU 한 장 + 2 GPU-days 수준으로 낮아진 훈련 장벽이 이 재훈련을 현실적인 선택지로 만든다는 점이 SuperGlue 시절과의 가장 큰 차이다.

## 한 줄 평 / 한계

- **평**: 새 문제를 연 논문이 아니라 SuperGlue라는 성숙한 시스템을 공학적으로 완성한 논문이다. "Sinkhorn을 double-softmax × matchability로 바꿔도 되고, 오히려 낫다"는 발견 하나와 "confidence로 깊이·폭을 조절한다"는 적응형 설계가 각각 독립적으로 가치 있으며, 결합하면 sparse matching의 실용 기본값이 바뀐다.
- **한계**: ① sparse 매처의 태생적 한계로 keypoint가 없는 곳은 못 잡는다 — MegaDepth-1800 공정 비교에서도 dense(LO-RANSAC AUC@5° 기준 약 2%)에 뒤진다. ② InLoc 실내에서는 강한 텍스처의 반복 물체(자판기 등)를 기하 구조 대신 매칭하는 실패 사례를 저자 스스로 보인다(Fig. 10). ③ 적응형 이득은 난이도 의존적이라 hard 쌍에서는 speedup이 1.16배로 줄어, 최악 latency 보장이 필요한 실시간 시스템이라면 이득을 보수적으로 잡아야 한다.
