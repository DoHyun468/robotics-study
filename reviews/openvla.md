# OpenVLA (2024)

*오픈소스 Vision-Language-Action 모델 — VLM을 로봇 정책으로*

Kim, Pertsch, Karamcheti 외, *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024 (arXiv 2406.09246). 카테고리 개요는 [VLA 리뷰 허브](vla.md), 실습 기록 전체는 [VLA 실습 페이지](../vla.md).

## 한 줄 요약

> Llama-2 7B + 이중 비전 인코더(DINOv2+SigLIP) 위에 로봇 action을 **언어 토큰처럼** 얹어, Open X-Embodiment 97만 궤적으로 모방학습한 **오픈 가중치 VLA**. 55B RT-2-X를 7분의 1 파라미터로 능가했고, LoRA·양자화로 "개인 GPU에서 만질 수 있는 VLA"의 기준선이 됐다. 우리가 LIBERO 4-suite를 직접 재현하고 자체 파인튜닝까지 돌린 모델이다.

---

## 1. 문제 — 왜 "로봇 정책"에 VLM을 쓰나

로봇 조작 정책은 보통 태스크·로봇(embodiment)마다 새로 학습된다. 이 방식은 두 가지가 아프다.

1. **데이터 비용**: 태스크 하나마다 시연을 새로 모아야 한다. 로봇 시연은 인터넷 텍스트처럼 긁어올 수 없다.
2. **일반화**: 학습 분포를 조금만 벗어나도(새 물체, 새 배경, 새 지시문) 무너진다.

한편 VLM(비전-언어 모델)은 인터넷급 사전학습 덕에 **물체·개념·언어 지시에 대한 일반화**를 이미 갖고 있다. 그렇다면 "이미지+지시문 → 텍스트"를 하는 VLM을 "이미지+지시문 → **action**"으로 바꿔 달면, VLM의 일반화를 로봇 제어에 이식할 수 있지 않을까 — 이것이 VLA(Vision-Language-Action)의 기본 아이디어다.

선행작 RT-2(-X)가 이 아이디어를 55B 규모로 증명했지만 **가중치가 비공개**였다. OpenVLA의 문제의식은 명확하다: **같은 성능을 더 작은 모델로, 그리고 완전한 오픈 가중치로** — 누구나 내려받아 자기 로봇에 파인튜닝할 수 있게.

## 2. 방법

### 2.1 아키텍처 — Prismatic VLM 골격

$$
\text{이미지 } o_t,\ \text{지시문 } \ell \;\longrightarrow\; \text{VLM} \;\longrightarrow\; \text{action 토큰 } a_t
$$

- **비전 인코더 2종 병렬**: **DINOv2**(공간·기하 특징에 강함)와 **SigLIP**(언어 정렬 의미 특징에 강함)의 patch feature를 **채널 방향으로 concat**. 기하와 의미를 동시에 잡으려는 설계로, 조작처럼 "어디를 어떻게 잡을지"가 중요한 태스크에서 단일 CLIP류 인코더보다 유리하다.
- **프로젝터**: 2-layer MLP가 시각 토큰을 언어 임베딩 공간으로 사상.
- **언어 백본**: **Llama-2 7B**. 시각 토큰 + 지시문 토큰을 받아 다음 토큰을 예측한다.

### 2.2 Action 토큰화 — 정책 학습을 language modeling으로

OpenVLA의 재미있는 지점. 연속 7-DoF action(Δ위치 3 + Δ회전 3 + 그리퍼 1)을 **차원별 256개 bin으로 이산화**한다.

- bin 경계는 학습 데이터의 **1~99 백분위 구간을 균등 분할** — 극단값(outlier)이 bin 폭을 망치는 걸 막는다.
- 이산화된 256개 값은 **Llama 토크나이저에서 사용 빈도가 가장 낮은 토큰 256개를 덮어써서** 어휘에 편입.

$$
a \in \mathbb{R}^7 \;\xrightarrow{\ \text{256-bin 양자화}\ }\; (k_1,\dots,k_7),\ k_i \in \{0,\dots,255\} \;\longrightarrow\; \text{7개의 "단어"}
$$

이제 정책 학습이 **그냥 다음-토큰 예측(behavior cloning)** 이 된다. 별도의 action head도, 회귀 손실도 없다 — VLM 학습 인프라를 그대로 재사용할 수 있다는 게 이 설계의 실용적 가치다.

### 2.3 학습

- **데이터**: Open X-Embodiment에서 큐레이션한 **97만 로봇 시연 궤적**(다양한 로봇·태스크·씬).
- **컴퓨트**: **A100 64장 × 14일 ≈ 21,500 A100-시간**, 27 epoch, lr 2e-5 (원문 명시).
- 학습 신호는 순수 모방(cross-entropy) — 보상도 RL도 없다. VLA 계열이 [world-model 기반 RL](../world-models/latent.md)과 갈라지는 지점이다(신호: 시연 vs 보상).

### 2.4 파인튜닝과 경량화 — "오픈"을 실질로 만드는 부분

- **LoRA**: rank **r=32**로 **전체 파라미터의 1.4%만** 학습해도 full fine-tuning과 동급 성능(원문). 새 태스크 적응이 **단일 A100 10~15시간**으로 줄어든다(약 8배 절감).
- **4-bit 양자화**: bfloat16 대비 메모리 **16.8GB → 7.0GB**, 성공률은 동급 — 소비자 GPU 추론이 현실이 된다.

## 3. 결과

### 3.1 원문 결과 (arXiv 2406.09246 대조)

| 평가 | OpenVLA (7B) | RT-2-X (55B) |
|------|--------------|---------------|
| BridgeData V2 (WidowX, 29 태스크) | **+16.5%p 절대 성공률 우위** | 기준 |
| Google robot 평가 | 동급 | 기준 |
| 파라미터 | 7B (**1/7**) | 55B |

- LoRA 1.4% 파라미터로 full FT 동급, 4-bit로 메모리 절반 이하 — §2.4.

### 3.2 우리 재현 (LIBERO 4-suite, 20 ep/suite)

| suite | 우리 실측 | 논문 보고 |
|-------|-----------|-----------|
| spatial | **80%** | 84.7% |
| object | **85%** | 88.4% |
| goal | **85%** | 79.2% |
| long | **45%** | 53.7% |

패턴이 정합한다(spatial~goal 높고 **long이 최난이도**). 평균 74%로 93M짜리 Octo(75.1%)와 거의 같다 — **LIBERO에선 7B의 20배 파라미터가 거의 안 드러난다**. 환경 셋업(WSL2 `ov` env, protobuf 삼각충돌·flash-attn·robosuite evdev 우회)은 [VLA 실습 페이지](../vla.md)에 기록.

## 4. 내 실습 연결

- 우리 [컨텍스트 사다리](../context.md)의 **L3**(언어+이미지→action; 인지·결정·IK를 정책이 통째로 대체)에 해당하는 모델을 직접 돌린 기록이다.
- **재현을 넘어 파인튜닝까지**: RLDS 변환→train→merge→eval의 자체 LoRA 파이프라인을 end-to-end 구동했다. GPU 시간 제약으로 돌린 bounded run(**1500 step ≈ 0.45 epoch**)은 LIBERO-Spatial **0%** — norm stats(action 역양자화 통계)를 직접 점검해 버그가 아니라 순수 **undertrain**임을 확인했다. 원문이 27 epoch을 돌린 걸 생각하면 당연한 결과인데, 직접 부딪혀보면 "체크포인트 실행해 논문값 재현"과 "VLA를 훈련시켜 쓸 만하게 만들기" 사이의 간격이 몸으로 확인된다(현재 lr 5e-4 다중 epoch 장기 run 진행 중).
- action 이산화(§2.2)는 우리 파이프라인 관점에서 보면 "연속 제어 문제를 분류 문제로 바꾸는" 트릭이다 — [DreamerV3](../world-models/dreamerv3.md)의 twohot, [MuZero](../world-models/muzero.md)의 categorical support와 같은 계열의 발상(연속 회귀를 이산 분포로)이 정책 출력단에서 반복된다.

## 5. 한 줄 평 / 한계

**한 줄 평.** "VLM을 로봇 정책으로"를 **오픈 가중치 + 개인 GPU 파인튜닝 가능**으로 민주화한 논문. 성능 주장(RT-2-X 대비 우위)보다도, 이후 모든 오픈 VLA 연구(Octo·OpenVLA-OFT·π0 계열 비교실험)의 **공통 기준선**이 됐다는 게 더 큰 기여다.

**한계.**
- **반응형(reactive) 정책**: 매 스텝 이미지→action 매핑일 뿐, 명시적 계획·상태 추정이 없다. long-horizon(LIBERO-long 45~54%)이 일관되게 약한 구조적 이유.
- **이산화 손실**: 256-bin 양자화는 정밀 조작에서 해상도 한계가 된다(후속 연구들이 diffusion head·연속 action으로 옮겨간 이유).
- **스케일 회의**: LIBERO에서 93M Octo와 동급 — **스케일이 항상 답은 아니다**. 사전학습 분포와 평가 분포가 가까울 때 대형 백본의 이점이 흐려진다.
- **추론 속도**: 7B 자기회귀 디코딩은 고주파 제어에 부담(단일 GPU에서 수 Hz 수준) — 양자화·병렬 디코딩이 후속 과제.
