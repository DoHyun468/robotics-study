# Trajectory Transformer (2021)

*궤적 전체를 하나의 언어로 — 상태·행동·보상을 전부 토큰화하고 beam search로 계획*

Janner, Li, Levine, *Offline Reinforcement Learning as One Big Sequence Modeling Problem* (Trajectory Transformer), NeurIPS 2021 (arXiv 2106.02039).

같은 갈래: [Decision Transformer](decision-transformer.md) · [시퀀스형 개요](sequence.md)

## 한 줄 요약

상태·행동·보상의 **모든 차원을 이산 토큰**으로 바꿔 궤적을 문자 그대로 "문장"으로 만들고, GPT 하나로 그 언어를 배운 뒤 **beam search**(return-to-go를 휴리스틱 삼아)로 좋은 미래를 골라 행동한다. 정책·가치·동역학 모델의 구분 자체를 지운 "one big sequence model"로, D4RL locomotion에서 CQL·[DT](decision-transformer.md)를 웃도는 평균 78.9를 기록 — 그리고 **장기 롤아웃에서 단일스텝 동역학 모델의 compounding error를 크게 이기는** 예측 품질을 보였다.

---

## 1. 문제 — 모델·정책·가치를 왜 따로 만들어야 하나

model-based RL의 표준 구성은 부품이 많다: 동역학 모델 + 보상 모델 + 정책 + 가치함수, 각각 다른 손실·다른 안정화 트릭. [Dreamer](dreamer.md)류가 대표적이다. 그런데 언어 모델링은 "무엇이 다음에 오는가"라는 단일 문제로 문법·의미·문맥을 전부 흡수한다.

> **TT의 질문**: 궤적 $(s_1, a_1, r_1, s_2, \dots)$도 결국 시퀀스인데, 동역학이든 정책이든 가치든 전부 "**다음 토큰 예측**" 하나로 배우면 안 되는가?

이게 되면 RL 알고리즘의 대부분이 사라지고, 남는 것은 (1) 토큰화 방법과 (2) 학습된 시퀀스 모델에서 **좋은 미래를 골라내는 디코딩 전략**뿐이다 — 언어 모델의 beam search가 RL의 계획(planning)이 된다.

## 2. 방법

### 2.1 토큰화 — 연속 궤적을 문장으로

상태 $s\in\mathbb{R}^N$, 행동 $a\in\mathbb{R}^M$, 보상 $r$의 **각 차원을 독립적으로 이산화**해 한 줄로 편다:

$$
\tau = \big(\,s_1^1,\dots,s_1^N,\; a_1^1,\dots,a_1^M,\; r_1,\; s_2^1,\dots\,\big) \qquad \text{(차원당 어휘 } V\text{개)}
$$

이산화 방식 두 가지를 비교(원문):

- **uniform**: 차원의 [min, max]를 $V$등분 — 유클리드 구조 보존, 대신 outlier가 구간을 낭비시킴.
- **quantile**: 데이터 분포의 등확률 구간으로 나눔 — 토큰마다 데이터 1/V씩, 빈 토큰이 안 생김.

[OpenVLA](../reviews/openvla.md)가 action을 1~99백분위 256-bin으로 자르는 것과 같은 문제의식이 3년 앞서 등장한 셈이다.

### 2.2 학습 — 그냥 GPT

토큰 열에 대한 표준 다음-토큰 예측(cross-entropy). 이 하나의 모델이 동시에:

- $p(s_{t+1} \mid s_t, a_t, \dots)$ — **동역학 모델**이고,
- $p(a_t \mid s_t, \dots)$ — 데이터 분포의 **정책**이고,
- $p(r_t, R_t \mid \dots)$ — **보상·가치 예측기**다.

부품 구분이 사라졌으므로 각 부품용 안정화 트릭(타깃 네트워크, 비관성 등)도 필요 없다.

### 2.3 계획 — beam search를 보상으로 유도

언어 모델의 beam search는 로그확률이 높은 문장을 찾는다. 이걸 그대로 쓰면 "그럴듯한" 미래만 나오지 "좋은" 미래가 아니다. TT의 수정: 시퀀스에 **return-to-go** $R_t=\sum_{t'\ge t}\gamma^{t'-t}r_{t'}$ 토큰을 포함시키고, beam 후보의 순위를 로그확률 대신 **누적 보상 + return-to-go 추정**으로 매긴다.

```
매 결정 시점:
  beam search로 후보 미래 시퀀스들을 전개 (토큰 단위 자기회귀 샘플)
  각 후보를 "지금까지 누적 보상 + 모델이 예측한 return-to-go"로 스코어링
  최고 후보의 첫 행동 실행 → receding horizon 반복
```

return-to-go가 **가치함수의 역할을 휴리스틱으로** 수행한다 — 지평 너머를 근시안적으로 자르지 않게 해주는 장치. ([PlaNet](planet.md)의 CEM이 잠재공간에서 하던 일을, TT는 토큰공간에서 beam search로 한다 — "학습된 모델 위의 MPC"라는 구조는 동일하다.)

## 3. 결과 (원문 대조)

**D4RL locomotion (Medium-Expert, 정규화 점수, quantile 이산화):**

| 태스크 | TT | CQL | [DT](decision-transformer.md) | MBOP |
|--------|-----|-----|-----|------|
| HalfCheetah | 95.0 | 91.6 | 86.8 | **105.9** |
| Hopper | **110.0** | 105.4 | 107.6 | 55.1 |
| Walker2d | 101.9 | **108.8** | 108.1 | 70.2 |
| 평균 (전체) | **78.9** | 77.6 | 74.7 | 47.8 |

**장기 예측 품질 (이 논문의 숨은 핵심)**: humanoid 100스텝 롤아웃에서, 표준 단일스텝 동역학 모델(PETS 앙상블)은 수십 스텝 만에 **compounding error로 물리적으로 불가능한 상태**를 만들지만, TT의 롤아웃은 실제 환경 궤적과 **육안 구분이 어려운** 수준을 유지(원문). 또 컨텍스트를 1스텝으로 자른 Markovian 버전과 비교하면 완전관측 태스크에선 비슷하지만 **부분관측에선 긴 컨텍스트가 확실히 이긴다** — 트랜스포머의 장거리 문맥이 world model의 이력 의존성 문제를 그대로 흡수한다는 증거다.

## 4. 스터디와의 개념적 연결

- **[DT](decision-transformer.md)와의 구도**: 같은 해, 같은 "RL=시퀀스" 명제의 두 답. DT는 **계획 없이 조건부 생성**(가볍고 빠름), TT는 **모델 기반 계획**(무겁지만 동역학 지식을 명시적으로 사용). 이 대비는 우리 파이프라인의 "반응형 정책 vs MPC" 선택과 정확히 같은 축이다.
- **compounding error 결과**는 이 사이트가 반복해 만나는 주제다 — [GameNGen](gamengen.md)은 노이즈 증강으로, [DIAMOND](diamond.md)는 EDM으로 픽셀공간의 표류를 눌렀는데, TT는 **긴 컨텍스트 + 토큰화**만으로 상태공간의 표류를 눌렀다. "자기회귀 롤아웃이 언제 무너지고 무엇이 지탱하는가"를 세 갈래로 비교해 읽을 만하다.
- 상태·행동·보상을 균일한 토큰으로 취급하는 설계는 이후 [Genie](genie.md)·[Cosmos](cosmos.md)의 AR world model, [OpenVLA](../reviews/openvla.md)의 action 토큰화로 이어지는 "모든 것은 토큰" 노선의 출발점이다.

## 5. 한 줄 평·한계

**한 줄 평.** "RL의 부품들을 전부 지우고 GPT 하나 + beam search로"를 실제 벤치 우위(78.9)와 장기 예측 품질로 뒷받침한 논문. DT가 발상의 전환이라면, TT는 그 발상이 **모델 기반 계획까지 포함해** 성립함을 보인 완결판이다.

**한계.**
- **추론 비용**: 매 결정마다 토큰 단위 beam search — 상태·행동 차원 수만큼 토큰이 늘어나 고차원·고주파 제어엔 무겁다(로봇 실시간 루프에 그대로는 부담).
- **이산화 손실**: 차원별 독립 이산화는 차원 간 상관을 토큰 수준에서 못 잡고, 정밀 연속 제어에선 해상도 한계가 된다.
- **오프라인 전제**: 데이터 분포가 계획의 상한 — beam search는 모델이 본 적 있는 미래 중에서만 고른다.
- **보상 라벨 필요**: return-to-go 토큰이 필수라, 보상 없는 시연 데이터엔 그대로 못 쓴다(그쪽은 [DT](decision-transformer.md)의 BC 모드 또는 [V-JEPA 2](vjepa2.md)류 목표 조건 계획이 답).
