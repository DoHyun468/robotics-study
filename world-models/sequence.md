# Sequence-Based World Models

RL을 "가치 학습·정책 최적화" 문제가 아니라 **시퀀스 모델링** 문제로 재정의하는 계열 — 궤적 $(s, a, r, \dots)$을 토큰 열로 보고 트랜스포머(GPT)로 다음 토큰을 예측한다. 명시적 world model(동역학 예측기)을 따로 두지 않거나(DT), 동역학·정책·가치를 **한 시퀀스 모델에 전부 흡수**한다(TT). 개념적 위치는 [Concepts](../concepts.md).

## 계보

```
GPT (언어)  ──→  Decision Transformer (2021)   행동 = return 조건 다음-토큰 예측
             └→  Trajectory Transformer (2021)  궤적 전체 = 하나의 언어, beam search로 계획
                          │
             (이후 VLA의 action 토큰화, Genie의 자기회귀 동역학으로 이어지는 발상의 원류)
```

- [Decision Transformer (2021)](decision-transformer.md) — return-to-go에 조건화한 행동 생성. 가치함수·부트스트래핑 없이 오프라인 RL.
- [Trajectory Transformer (2021)](trajectory-transformer.md) — 상태·행동·보상 전부를 토큰화해 한 GPT로 학습, beam search로 계획.

## 비교표

| 모델 | 연도 | 궤적을 무엇으로 보나 | 행동 결정 | 대표 수치 (원문 대조) |
|------|------|---------------------|-----------|----------------------|
| [Decision Transformer](decision-transformer.md) | 2021 | (R̂, s, a) 반복 시퀀스 — R̂=return-to-go | 원하는 return을 조건으로 **다음 행동 토큰 생성** (계획 없음) | D4RL 평균 69.2 vs CQL 54.2; 지연 보상에서 107.3 vs CQL 9.0 |
| [Trajectory Transformer](trajectory-transformer.md) | 2021 | 상태·행동·보상 **전 차원을 이산 토큰화**한 하나의 열 | **beam search** (return-to-go 휴리스틱으로 유도) | D4RL locomotion 평균 78.9 vs CQL 77.6·DT 74.7 |

## 이 갈래가 world model 지도에서 갖는 의미

- **"RL의 3요소(정책·가치·모델)를 트랜스포머 하나가 대체할 수 있다"**는 발상의 원류. TT의 저자 표현대로 "big sequence model이 곧 정책이자 모델"이다.
- 부트스트래핑(TD)의 불안정을 지도학습(다음 토큰 예측)으로 우회 — 특히 **희소·지연 보상**에서 TD 기반이 무너질 때 강하다(DT의 핵심 실증).
- 이후 계보와의 연결: action을 토큰으로 다루는 발상은 [OpenVLA](../reviews/openvla.md)의 action 토큰화로, 자기회귀 시퀀스 생성으로 세계를 굴리는 발상은 [Genie](genie.md)·[Cosmos](cosmos.md)의 AR world model로 이어진다. Dreamer류([Latent](latent.md))와 달리 **상상 롤아웃으로 정책을 개선하지 않고**, 데이터에 있는 궤적 분포를 조건부로 재생한다는 게 근본 차이 — 그래서 오프라인 RL에 자연스럽고, 데이터 분포 밖 개선에는 약하다.
