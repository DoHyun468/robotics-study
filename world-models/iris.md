# IRIS (2023)

*프레임을 16개 토큰 문장으로 — GPT world model 안에서 배우는 Atari 에이전트*

Micheli, Alonso, Fleuret, *Transformers are Sample-Efficient World Models* (IRIS), ICLR 2023 oral (arXiv 2209.00588).

같은 갈래: [Decision Transformer](decision-transformer.md) · [Trajectory Transformer](trajectory-transformer.md) · [시퀀스형 개요](sequence.md)

## 한 줄 요약

관측 프레임을 VQ 이산 토큰 **16개**로 바꾸고, **GPT 스타일 트랜스포머가 다음 프레임 토큰·보상·종료를 자기회귀로 예측**하는 world model을 만든 뒤, [Dreamer](dreamer.md)식으로 그 상상 안에서 정책을 배운다. Atari 100k(실플레이 2시간 분량)에서 **mean HNS 1.046, 26게임 중 10게임 초인간** — "트랜스포머 시퀀스 모델이 곧 world model"임을 샘플효율 벤치에서 보인, [DT/TT](decision-transformer.md)의 시퀀스 발상과 Dreamer의 상상 학습을 잇는 다리다.

---

## 1. 문제 — RNN 잠재 대신 토큰 시퀀스로 세계를 배우면?

[DreamerV2](dreamerv2.md)까지의 world model은 **RNN(RSSM) + 압축 잠재**가 표준이었다. 한편 언어 쪽에선 트랜스포머의 다음-토큰 예측이 장거리 구조 학습의 표준이 됐고, [Trajectory Transformer](trajectory-transformer.md)는 저차원 궤적에서 "RL 전부를 시퀀스 모델링으로"가 됨을 보였다. 남은 질문:

> **픽셀 관측도 토큰으로 바꾸면, 트랜스포머가 RSSM보다 나은 world model이 되는가?** — 특히 데이터가 아주 적을 때(샘플효율).

Atari 100k(환경 스텝 10만 = 사람 플레이 약 2시간)가 정확히 이걸 재는 벤치다.

## 2. 방법 — 세 부품

### 2.1 이산 오토인코더 — 프레임을 "단어 16개"로

VQ-VAE(지각 손실 포함)로 64×64 프레임을 **토큰 16개**(어휘 512)로 압축한다. 프레임이 "16단어 문장"이 되는 셈 — [Genie](genie.md)의 tokenizer, [Cosmos](cosmos.md)의 discrete tokenizer와 같은 노선의 이른 사례다.

### 2.2 GPT world model — 다음 프레임을 자기회귀로

토큰화된 이력을 받아 세 가지를 예측한다:

$$
\big(o^{1:16}_1, a_1,\; o^{1:16}_2, a_2, \dots\big) \;\xrightarrow{\ \text{GPT}\ }\; \hat o^{1:16}_{t+1}\ (\text{자기회귀}),\quad \hat r_t,\quad \hat d_t\ (\text{종료})
$$

구성(원문): 타임스텝 $L=20$, 임베딩 256, 레이어 10, 헤드 4 — 언어 모델 기준으론 아주 작다. RSSM의 "결정적+확률적 잠재" 설계 대신, **불확실성은 토큰 위 categorical 분포가, 장기 의존은 attention이** 자연스럽게 담당한다.

### 2.3 상상 학습 — Dreamer 골격 그대로

world model이 곧 시뮬레이터: 실데이터 상태에서 출발해 **지평 $H=20$**의 상상 롤아웃을 GPT로 생성하고, 그 안에서 actor-critic(LSTM 정책, λ-return $\lambda=0.95$, $\gamma=0.995$)을 학습한다. 학습 루프는 [Dreamer](dreamer.md) 2.4절의 3단 루프(수집→모델→상상)와 동일 — **바뀐 건 world model의 몸통뿐**이라, "RSSM vs 토큰 트랜스포머"의 비교가 깨끗해진다.

## 3. 결과 (원문 대조)

**Atari 100k, 26게임:**

| 방법 | mean HNS | 비고 |
|------|----------|------|
| **IRIS** | **1.046** | median 0.289, IQM 0.501, **10/26게임 초인간** |
| SPR | 0.616 | model-free 대표 (IRIS가 +70%) |
| SimPLe | 0.332 | 픽셀 예측 WM 선행작 |
| DrQ / CURL | 0.261 | 증강 기반 model-free |
| EfficientZero | 1.943 | 단, **lookahead search 사용** — 탐색 없는 비교군이 아님 |

탐색(search) 없이 상상 학습만 쓰는 방법 중 당시 최고 — 이후 [DIAMOND](diamond.md)(1.46)가 같은 벤치에서 확산 생성으로 갱신하는 흐름으로 이어진다(IRIS의 이산 토큰이 뭉개는 디테일을 확산이 보존한다는 게 DIAMOND의 출발점이었음을 상기).

## 4. 스터디와의 개념적 연결

- **계보의 다리**: [TT](trajectory-transformer.md)가 저차원 궤적을, IRIS가 픽셀 관측을 토큰화했다 — "모든 것은 토큰" 노선이 시각 도메인으로 넘어온 지점이고, 이 노선의 끝에 [Genie](genie.md)(10.7B)·[Cosmos](cosmos.md)(AR 계열)가 있다. 시퀀스형 허브의 세 논문을 "무엇을 토큰화했나"로 비교하면: TT=상태·행동·보상, DT=return 조건 행동, IRIS=**관측 그 자체**.
- **DreamerV2와의 대조 실험**으로 읽는 가치: 같은 상상 학습 루프에서 world model만 RSSM→GPT로 바꾼 구도라, "시퀀스 모델이 world model로서 갖는 득실"(장거리 attention vs 추론 비용)이 분리돼 보인다.
- 16토큰이라는 강한 압축은 [DIAMOND](diamond.md)가 지적한 "디테일 소실 → 정책 손해" 문제를 그대로 안고 있다 — 두 리뷰를 나란히 읽으면 압축-충실도 트레이드오프의 양끝이 잡힌다.

## 5. 한 줄 평·한계

**한 줄 평.** "world model을 언어 모델처럼" — 프레임을 문장으로 바꾸고 GPT에게 세계를 가르친 뒤 그 꿈에서 정책을 배운다. 발상의 단순함과 Atari 100k 결과로, 토큰 자기회귀가 world model의 유력한 몸통임을 각인시킨 논문.

**한계.**
- **토큰 압축의 디테일 소실**: 16토큰/프레임은 작은 물체(총알·공)를 뭉갤 수 있다 — 정확히 이 지점을 [DIAMOND](diamond.md)가 공격해 1.46으로 갱신했다.
- **자기회귀 비용**: 프레임당 16토큰 순차 생성 × 상상 지평 20 — RSSM 한 스텝보다 무겁고, 해상도를 올리면 토큰 수가 폭증한다.
- **중앙값의 그늘**: mean 1.046 대비 median 0.289 — 소수 게임의 대승이 평균을 끌어올린 분포라, "전반적 초인간"과는 거리가 있다(원문도 IQM·optimality gap을 함께 보고).
- **Atari 한정**: 연속제어·실로봇으로의 확장은 이 논문 범위 밖 — 그쪽 답은 [TD-MPC](tdmpc.md)·[Dreamer](dreamer.md) 계열이 쥐고 있다.
