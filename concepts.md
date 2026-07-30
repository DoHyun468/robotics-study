# Concepts — RL · World Models

이 프로젝트의 데모는 대부분 **모방(시연→action)** 또는 **고전 기하 파이프라인**이다. 그 반대편 축 — **보상으로 스스로 배우는 model-based RL**과 **world model** — 을 개념적으로 정리한다. "OpenVLA 같은 VLA와 Dreamer/MuZero는 뭐가 다른가", "world model은 지금 어디까지 왔나", "전직을 위해 RL을 따로 파야 하나"라는 질문에 답하는 페이지.

## 1. Model-based RL / World Model 계열

핵심 아이디어는 하나다: **환경의 동역학(다음 상태·보상)을 모델로 배우고, 그 모델 안에서 계획하거나 "상상"하며 정책을 최적화한다.** 실환경 상호작용이 비싼 로봇·게임에서 샘플 효율을 끌어올리는 길.

| 모델 | 발표 | world model | 정책/계획 | 한 줄 |
|---|---|---|---|---|
| **PlaNet** | 2019 | RSSM(latent dynamics) | **CEM 온라인 계획** | 픽셀에서 latent 동역학을 배워 그 안에서 계획(정책망 없음) |
| **Dreamer V1** | 2020 | RSSM | 상상 롤아웃 + actor-critic | latent 상상 궤적에서 정책 학습(연속제어 DMC) |
| **Dreamer V2** | 2021 | 이산 latent RSSM | 상상 actor-critic | Atari에서 인간급, 단일 GPU |
| **Dreamer V3** | 2023 | RSSM(+symlog 등 안정화) | 상상 actor-critic | **하이퍼파라미터 하나로 150+ 태스크**, Minecraft 다이아몬드 from scratch |
| **MuZero** | 2020 | 잠재 동역학(보상·가치·정책만 예측) | **MCTS 탐색** | 규칙을 몰라도 Atari·바둑·장기 SOTA |

- **PlaNet → Dreamer** 계보: PlaNet은 RSSM으로 동역학만 배우고 매 스텝 CEM으로 행동을 최적화. Dreamer는 그 위에 **actor-critic을 얹어**, 실데이터가 아니라 world model이 생성한 **상상 latent 궤적**에서 정책을 학습 → 훨씬 효율적. V1(연속)→V2(이산 latent, Atari)→V3(도메인 불문 견고 스케일링).
- **MuZero**는 결이 조금 다르다. 관측을 재구성하는 완전한 world model 대신, **계획에 필요한 것(보상·가치·정책)만** 예측하는 잠재 모델을 배우고 **MCTS**로 탐색. AlphaZero를 "규칙을 모르는" 환경으로 확장.

공통점: **동역학 학습 → 계획/상상으로 정책 최적화, 신호는 보상.** 이게 우리 데모의 모방/고전 파이프라인과 대비되는 지점.

## 2. VLA vs World-model RL — 왜 직접 비교가 안 되나

OpenVLA와 Dreamer를 "누가 더 세냐"로 붙이기 어렵다. **다른 문제를 다른 신호로** 푼다.

| | OpenVLA (VLA) | Dreamer / MuZero (WM-RL) |
|---|---|---|
| 패러다임 | 대규모 **모방(behavior cloning)** | model-based **강화학습** |
| 학습 신호 | 전문가 **시연** | **보상** + 자기 경험 |
| world model | 없음(관측→행동 반응형) | **있음**(동역학을 명시적으로 배움) |
| 언어 | 언어 조건부 · VLM 사전학습 | 보통 없음 |
| 추론 | 1-pass forward(빠름) | MCTS 탐색(MuZero) / 반응형(Dreamer) |
| 대표 벤치 | LIBERO, Open X-Embodiment | Atari, DMControl, Go, Minecraft |
| 일반화 원천 | 인터넷급 VLM + 대규모 로봇 시연 | 환경 내 자기경험 |

- VLA는 **"많이 본" 것을 흉내** 낸다 — 언어 일반화와 다양한 물체·태스크에 강하지만, 보상으로 새 전략을 발견하진 않는다.
- WM-RL은 **"많이 해본" 것에서 배운다** — 보상 설계와 온라인 상호작용이 필요하지만, 시연에 없던 전략도 계획으로 찾아낸다.
- 그래서 LIBERO 성공률(OpenVLA)과 Atari 점수(Dreamer)를 같은 표에 못 놓는다. 우리 [VLA 페이지](vla.md)의 76.5% 같은 수치는 **모방 벤치**의 값이다.

## 3. World model 3축 (성숙도 현황)

"world model"이 한 덩어리가 아니다. 목적에 따라 세 갈래이고 성숙도가 다르다.

- **예측·표현형** (Dreamer, JEPA/V-JEPA, MuZero): 제어·계획·샘플효율에서 가장 **성숙**. latent 공간에서 미래를 예측/평가. **V-JEPA-2가 실제 로봇 계획에 쓰이기 시작** — 로봇 실용의 최전선. 3D/기하 강점과 붙이기 좋은 갈래.
- **시퀀스·토큰형** (Decision/Trajectory Transformer, IRIS, Genie 계열): 궤적·관측을 토큰으로 만들어 Transformer로 다음-토큰 예측. 샘플효율↑, **인터랙티브 환경 생성**이 부상. RL을 시퀀스 모델링으로 재정의한 원류(DT·TT)는 [Sequence-Based 리뷰](world-models/sequence.md) 참고.
- **생성·영상형** (DIAMOND, GameNGen, Oasis, Cosmos, Genie-3): 화제성·스케일 최고, 보기엔 화려. 그러나 **장기 물리 일관성·정확한 장기 예측·정밀 제어가 미해결** → 당장 로봇 제어기로 쓰기보다 **데이터/시뮬레이션 생성**(sim 자산, 도메인 랜덤화)에 먼저 유용.

요지: **로봇 제어에 실제로 닿아 있는 건 예측·표현형**(특히 JEPA류)이고, 생성·영상형은 인상적이지만 제어 신뢰성은 아직이다.

→ 각 모델의 수식·수치·알고리즘 심화는 [World Models](world-models/latent.md) 트랙에서 논문 원문 대조로 정리했다 — [Latent](world-models/latent.md)(World Models 2018·PlaNet·Dreamer V1/V2/V3·MuZero·TD-MPC) · [Sequence](world-models/sequence.md)(DT·TT·IRIS) · [Predictive](world-models/predictive.md)(V-JEPA 1/2) · [Generative](world-models/generative.md)(GAIA-1·GameNGen·Genie·DIAMOND·Cosmos).

## 4. 공부 방향 메모 (개인)

"RL을 따로 파야 하나, world model만 파야 하나, 뭐가 제일 빠른가"에 대한 개인 결론:

- **RL은 언어 수준만** — MDP, policy/value, model-free vs model-based, BC vs RL의 차이를 개념으로 이해하면 충분. 풀스택 RL(보상 설계·안정화·대규모 튜닝)은 타깃 포지션(perception / spatial AI / VLA)에 **미스핏**이라 투자 대비 효율이 낮다.
- **world model은 예측형(JEPA)**을 3D/재구성 강점 위에 얹고, 생성형은 sim/데이터 생성 관점으로 훑는다.
- 무엇보다 **perception → action 데모를 계속 출하**하는 것(이 사이트의 [perception](perception.md)·[manipulation](manipulation.md)·[grasp A/B](grasp_sota.md)·[VLA 재현](vla.md))이 전직 최단거리. 개념 정리는 그 데모를 설명하는 언어로만 쓰면 된다.

*이 페이지는 논문/개념의 정성적 정리이며, 이 프로젝트에서 직접 학습·측정한 수치가 아니다(측정값은 다른 페이지 참조).*
