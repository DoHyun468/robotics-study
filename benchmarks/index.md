# Benchmarks & Task Suites — RL·조작 벤치 지도

논문 Figure에서 "DMControl 39 tasks", "Meta-World 50 tasks" 같은 막대가 나올 때, 그 이름들이 **무엇이고 서로 어떻게 다른지**를 정리하는 층이다. 모델 리뷰([world models](../world-models/latent.md)·[VLA](../reviews/vla.md))가 "방법"을 다룬다면, 여기는 그 방법들이 **채점되는 운동장**을 다룬다.

## 0. 용어부터 — 태스크·스위트·벤치마크·데이터셋은 다른 말이다

| 용어 | 뜻 | 예 |
|---|---|---|
| **태스크(task)** | 보상 또는 성공 판정이 정의된 문제 하나 | `cheetah-run`, `door-open`, `PickCube` |
| **도메인/embodiment** | 태스크가 올라타는 몸·환경 | cheetah(몸 하나에 run·jump 등 여러 태스크) |
| **태스크 스위트 = 벤치마크** | 태스크 묶음 + 표준 평가 프로토콜 | DMControl, Meta-World, ManiSkill, MyoSuite, LIBERO |
| **데이터셋** | 고정된 (관측, 행동, 보상…) 기록 | D4RL(오프라인 RL), Open X-Embodiment(모방), LIBERO 시연 |
| **시뮬레이터** | 스위트가 얹히는 물리 엔진 | MuJoCo, SAPIEN, robosuite/MuJoCo |

혼동의 핵심을 짚으면 — **온라인 RL 벤치는 데이터셋이 아니다.** DMControl·Meta-World는 에이전트가 직접 상호작용하는 **환경 모음**이고, 점수는 "그 환경에서 배워서 얼마나 잘하나"다. 반대로 D4RL·OXE는 **기록된 데이터**이고, 점수는 "이 데이터만 보고 얼마나 잘하나"다. LIBERO는 둘 다다 — 환경(벤치)이면서 사람 시연 데이터셋을 함께 배포한다. 어떤 이름이 벤치인지 데이터셋인지 헷갈리면 "**에이전트가 env.step()을 부르는가**"를 물으면 된다.

## 1. 사례로 읽기 — TD-MPC2 Figure 1이 말하는 것

[TD-MPC2](../world-models/tdmpc.md) 논문의 Figure 1(arXiv 2310.16828)은 이 층의 용어가 전부 등장하는 좋은 독해 연습이다. 원문 대조 기준:

**오른쪽 (Single-task, 6패널)** — 태스크마다 **별도 에이전트**를 학습시킨 성적. 패널 = 스위트(또는 그 서브셋):

| 패널 | 정체 | 태스크 수 |
|---|---|---|
| [DMControl](dmcontrol.md) | 연속제어 표준 스위트(원본 19 + 커스텀 등 총 39) | 39 |
| [Meta-World](metaworld.md) | Sawyer 팔 조작 50태스크 | 50 |
| [ManiSkill2](maniskill.md) | SAPIEN 기반 조작 | 5 |
| Locomotion | **DMControl의 서브셋** — Humanoid·Dog 고차원 보행체 | 7 |
| [MyoSuite](myosuite.md) | 근골격(근육 구동) 제어 | 10 |
| Pick YCB | **ManiSkill2의 태스크 하나** — YCB 74개 물체 집기 | 1 |

즉 6패널이 6개의 다른 벤치가 아니다 — 스위트 4개(39+50+5+10 = **104태스크**) + 그중 어려운 서브셋 2개(Locomotion, Pick YCB)를 따로 조명한 것. 세로축 normalized score는 Meta-World류 = **성공률**, DMControl = **return을 [0,100]로 정규화**한 값의 평균이다(스위트마다 채점 단위가 다르므로 정규화 없이는 평균을 못 낸다 — 벤치 독해의 기본기).

**왼쪽 (Multi-task)** — **에이전트 하나**를 80태스크(Meta-World 50 + DMControl 30)에 동시에 학습시키고, 모델 크기를 1M→5M→19M→48M→317M로 키운 스케일링 곡선. TD-MPC(파랑)는 커질수록 **떨어지고**, TD-MPC2(빨강)는 단조 상승 — "월드모델 RL도 단일 하이퍼·단일 에이전트로 스케일이 된다"가 이 논문의 헤드라인 주장이고, 오른쪽은 "그 견고성이 싱글태스크에서도 성립한다"(SAC·DreamerV3 대비)는 보조 증거다.

우리 트랙 연결: [W6에서 TD-MPC2 공식 체크포인트(cheetah-run)를 직접 로드해 return 863±12를 재현](../wm.md)하고 내부 latent world model의 horizon drift를 실측했다 — 이 Figure의 DMControl 막대 하나를 우리 손으로 확인한 셈.

## 2. 이 층의 페이지

- [DMControl](dmcontrol.md) — 연속제어의 "공용 자"(1000스텝·보상 [0,1] 규격화). 이미지: **자체 렌더**.
- [Meta-World](metaworld.md) — 조작 50태스크 + 멀티태스크/메타러닝 프로토콜(MT/ML).
- [ManiSkill](maniskill.md) — SAPIEN 위 대규모 조작(20 태스크 패밀리·2000+ 물체·시연 4M+ 프레임).
- [MyoSuite](myosuite.md) — 토크가 아니라 **근육**으로 구동하는 제어(39근육 손).
- [LIBERO](libero.md) — 언어 조건 조작 벤치(130태스크) — **우리가 OpenVLA로 직접 돌린** 곳.

## 3. 한 장 비교표

| 스위트 | 시뮬레이터 | 규모 | 제어 대상 | 채점 | 주 사용처(우리 리뷰 기준) |
|---|---|---|---|---|---|
| [DMControl](dmcontrol.md) | MuJoCo | ~20 도메인·수십 태스크 | 추상 보행체·간단 몸 | return (max 1000) | [PlaNet](../world-models/planet.md)·[Dreamer](../world-models/dreamer.md)·[TD-MPC](../world-models/tdmpc.md) |
| [Meta-World](metaworld.md) | MuJoCo | 50 태스크 | Sawyer 팔 + 탁상 물체 | 성공률(거리 임계) | TD-MPC2 멀티태스크·메타RL 계열 |
| [ManiSkill](maniskill.md) | SAPIEN | 20 패밀리·2000+ 물체 | 팔(고정·모바일)·양팔 | 성공률 | 조작 RL/IL, TD-MPC2의 Pick YCB |
| [MyoSuite](myosuite.md) | MuJoCo | 3 모델·9 태스크군 | **근골격**(근육 구동) | 성공률/자세 오차 | 근육 제어 RL — TD-MPC2 평가축 |
| [LIBERO](libero.md) | robosuite(MuJoCo) | 130 태스크 | Franka + 언어 지시 | 성공률(BDDL 술어) | [OpenVLA](../reviews/openvla.md)·[π0](../reviews/pi0.md) 등 VLA |

## 4. 여기 없는 이름들 — 지도 완성용 한 줄씩

- **Atari (ALE)**: 픽셀 입력·**이산 행동** 게임 57종 — [DreamerV2](../world-models/dreamerv2.md)·[MuZero](../world-models/muzero.md)의 무대. **Atari 100k**는 같은 게임을 "10만 스텝(≈2시간)만 상호작용" 제한으로 채점하는 **샘플효율 벤치**([IRIS](../world-models/iris.md)·[DIAMOND](../world-models/diamond.md)).
- **D4RL**: 오프라인 RL **데이터셋** 모음(locomotion·미로 등) — 환경이 아니라 기록으로 배운다. [Decision Transformer](../world-models/decision-transformer.md)·[TT](../world-models/trajectory-transformer.md)의 채점장.
- **Open X-Embodiment(OXE)**: 다로봇 **시연 데이터셋**(~1M 궤적) — [OpenVLA](../reviews/openvla.md) 사전학습의 원료. 벤치가 아니라 공급망.
- **Crafter·Minecraft·ProcGen**: 절차 생성·장기 과제 축 — [DreamerV3](../world-models/dreamerv3.md)의 "150+ 태스크"를 구성하는 나머지 도메인들.

**읽는 감각 하나로 마무리**: 벤치 이름 옆의 숫자(39, 50, 104…)는 난이도가 아니라 **커버리지**다. 진짜 정보는 (1) 관측이 상태인가 픽셀인가, (2) 행동이 연속인가 이산인가, (3) 채점이 return인가 성공률인가, (4) 상호작용 예산이 얼마인가 — 이 네 축이 같아야 두 논문의 막대를 나란히 읽을 수 있다.
