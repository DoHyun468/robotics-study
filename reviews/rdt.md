# RDT-1B (2024)

*양팔(bimanual) 로봇을 위한 diffusion 파운데이션 모델 — 이산 토큰 대신 연속 action을 직접 생성*

Liu, Wu, Li, Tan, Chen, Wang, Xu, Su, Zhu (Tsinghua University, TSAIL 연구실), *RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation*, 2024 (arXiv 2410.07864). 카테고리 개요는 [VLA 리뷰 허브](vla.md). **이 리뷰는 문헌 리뷰다** — [OpenVLA](openvla.md)처럼 직접 재현·파인튜닝을 돌리지 않았고, 원문·프로젝트 페이지·공식 저장소를 근거로 정리했다. 양팔 협응은 우리 [매니퓰레이션](../manipulation.md) 실습(단일 Franka 팔, 고전 파이프라인) 범위 밖의 문제다.

## 한 줄 요약

> 두 팔 협응이 만드는 multi-modal action 분포를 **diffusion**으로 직접 모델링하는 1.2B 파라미터 트랜스포머. 46개 멀티로봇 데이터셋·100만+ 궤적으로 사전학습한 뒤, 자체 수집한 ALOHA 양팔 6천+ 에피소드로 파인튜닝했다. 이종 로봇의 action·상태 공간을 **물리적 의미를 보존한 채 통합**하는 "Physically Interpretable Unified Action Space"가 데이터 희소성 문제에 대한 핵심 해법이고, 가중치·코드·데이터셋을 전부 공개했다.

---

## 1. 문제 — 왜 양팔 파운데이션 모델이 어려운가

양팔(bimanual) 조작은 두 가지 축에서 단일팔보다 근본적으로 어렵다.

1. **Multi-modal action 분포**: 같은 상태에서도 두 팔이 협응하는 방식은 여러 가지가 유효하다(예: 어느 팔이 먼저 움직이는지, 물체를 어느 손에서 어느 손으로 넘기는지). 회귀(L2 손실) 기반 action head는 이런 다봉분포를 평균으로 뭉개버리는 mode collapse에 취약하고, [OpenVLA](openvla.md)식 256-bin 이산화도 차원마다 독립적으로 토큰화하면 두 팔 사이의 상관관계(joint distribution)를 놓친다.
2. **데이터 희소성**: 로봇 시연 데이터의 절대다수(Open X-Embodiment 등)는 단일팔이다. 양팔 데이터는 훨씬 적고 수집 비용도 크다. 대규모 사전학습으로 이 격차를 메우려면 서로 다른 로봇(단일팔·양팔·관절 제어·엔드이펙터 제어·바퀴형 이동체)의 이종 action 공간을 하나의 모델로 함께 학습해야 하는데, 단순히 최대 차원에 맞춰 zero-padding하면 "3번째 차원"이 로봇 A에서는 그리퍼 폭, 로봇 B에서는 팔꿈치 관절각을 의미하는 **의미 충돌**이 생겨 사전학습이 서로를 방해한다.

RDT의 문제의식은 이 둘을 동시에 겨냥한다: **diffusion으로 다봉분포를 표현력 있게 모델링**하고, **물리적 의미를 보존하는 통합 action 공간으로 이종 로봇 사전학습을 가능하게** 만드는 것.

## 2. 방법

### 2.1 왜 diffusion인가 — 연속 action을 직접 denoise

$$
a_t^{k-1} \;=\; \text{Denoise}_\theta\!\left(a_t^{k},\, k \mid o_t,\, \ell,\, s_t\right)
$$

OpenVLA가 연속 7-DoF action을 256-bin으로 이산화해 "다음 토큰 예측"으로 바꿨다면, RDT는 반대 극단이다. action chunk(연속 벡터 시퀀스)에 노이즈를 씌운 뒤, 이미지 $o_t$·언어 $\ell$·현재 상태 $s_t$·diffusion timestep $k$를 조건으로 이를 반복적으로 denoise한다. 이산화 없이 연속 분포를 직접 표현하므로 다봉분포·고차원 joint action(양팔 14+ DoF)에서 해상도·상관관계 손실이 없다. 논문은 이를 "가장 큰 diffusion 기반 로봇 조작 파운데이션 모델"(1.2B)이라 표현한다.

### 2.2 아키텍처 — RDT 트랜스포머 블록

- **백본**: Diffusion Transformer(DiT) 계열을 1.2B로 스케일업.
- **비전/언어 인코더**: **SigLIP**(이미지, 최대 3개 카메라 뷰), **T5-XXL**(언어) — 둘 다 **동결(frozen)**.
- **저차원 입력**(proprioception, 노이즈 낀 action, 제어 주파수, diffusion timestep)은 Fourier feature + MLP로 인코딩.
- **QKNorm**: attention 계산 시 수치 불안정을 방지.
- **RMSNorm**: LayerNorm 대신 사용해 시계열 데이터에서 토큰·attention이 튀는 현상을 완화.
- **비선형 MLP 디코더**: 선형 projection 대신 사용해 로봇 action의 비선형성을 더 잘 근사.
- **Alternating Condition Injection (ACI)**: 이미지·언어 조건을 매 층에 동시에 주입하지 않고 층마다 번갈아 cross-attention으로 주입 — 이미지 토큰이 언어 정보를 압도하는 것을 방지.
- 학습 중 **입력 modality를 무작위 마스킹**해 특정 modality(예: 이미지)에 과의존하지 않도록 함.
- action은 **한 번에 chunk 단위**(공식 구현 기준 64-step)로 예측 — ACT·Diffusion Policy와 같은 action chunking 철학을 diffusion에 얹은 것.

### 2.3 Physically Interpretable Unified Action Space — 데이터 희소성에 대한 정면 대응

이종 로봇의 action·상태를 억지로 같은 저차원 잠재공간에 눌러 담는 대신, **물리량 슬롯**(x/y/z 위치, 여러 회전 표현, 그리퍼 개폐 폭, 관절각 인덱스, 바퀴형 이동체의 base 속도 등)으로 이뤄진 공유 벡터를 정의하고, 각 로봇의 원래 action 차원을 해당하는 물리적 의미의 슬롯에 매핑한 뒤 나머지는 0으로 패딩한다. action $a_t$가 곧 다음 시점의 목표 상태 $s_{t+1}$의 부분집합이라는 관찰에 따라, **action과 proprioception이 같은 통합 공간을 공유**한다. 이 설계 덕분에 46개 서로 다른 데이터셋을 같은 출력 차원에서 학습해도 물리적으로 다른 의미가 충돌하지 않는다 — "스케일이 큰 사전학습이 오히려 서로를 방해하는" 문제를 구조적으로 막는 장치다.

### 2.4 학습 — 사전학습 → 파인튜닝

- **사전학습**: 46개 멀티로봇 데이터셋(Open X-Embodiment 포함), **100만+ 궤적**, H100 80GB **48장으로 약 한 달**, **총 100만 스텝**. 손실은 표준 DDPM denoising MSE.
- **파인튜닝**: **ALOHA 양팔 로봇**에서 자체 수집한 **6천+ 에피소드**(300개 이상 태스크, 100개 이상 물체, 15개 이상 방/조명 조건, 언어 지시문은 GPT-4-Turbo로 증강). 동일 H100 48장으로 **13만 스텝, 약 3일**.
- **추론**: 학습은 표준 다단계 diffusion이지만, 추론 시 **DPM-Solver++로 5-step까지 압축** — RTX 4090 24GB에서 **action chunk 6Hz**, 개별 action 기준 **약 381Hz**로 실시간 폐루프 제어가 가능한 속도를 낸다.
- 가중치(`robotics-diffusion-transformer/rdt-1b`, HuggingFace)·코드(`thu-ml/RoboticsDiffusionTransformer`, GitHub)·데이터셋을 전부 공개.

## 3. 결과

### 3.1 원문 결과 (arXiv 2410.07864 Table 3 대조, 자체 수집 ALOHA 태스크 기준)

| 평가 축 | RDT-1B | ACT | OpenVLA | Octo |
|---|---|---|---|---|
| 미지 물체 (Wash Cup, seen cup) | **87.5%** | 37.5% | 0% | 0% |
| 미지 씬 (Pour Water, unseen room) | **62.5%** | 12.5~37.5% | 0% | 12.5% |
| 지시문 따르기 (Pour Water 양 조절) | **87.5~100%** | — | 0~50% | 0% |
| Few-shot: Handover (5-shot) | **100%** | 88% | 84% | — |
| Few-shot: Fold Shorts (1-shot) | **68%** | — | — | 4% |
| 정교 조작 (Robot Dog, 걷기/조이스틱) | **48% / 76%** | 32% | 0% | 0% |

논문 기준으로 미지 물체·미지 씬·언어 지시·few-shot·정교 조작 전 축에서 RDT가 ACT·OpenVLA·Octo를 크게 앞선다 — 특히 OpenVLA·Octo는 여러 축에서 **0%**로, ALOHA 양팔 태스크가 원래 사전학습 분포(주로 단일팔)와 멀어 일반화가 거의 안 되는 것으로 보고된다.

### 3.2 후속 controlled 비교에서 드러난 결이 다른 그림

원문 수치는 **저자 자신의 기본 파인튜닝 레시피** 기준 비교다. OpenVLA-OFT 논문(Kim 외, arXiv 2502.19645)이 동일 ALOHA 실물 태스크에서 π0·RDT-1B·ACT·Diffusion Policy를 **통일된 조건**으로 다시 비교했는데,

- OpenVLA-OFT가 RDT-1B·π0·ACT·Diffusion Policy를 평균 성공률 기준 **최대 15%p(절대치)** 앞섰고,
- **Diffusion Policy가 옷 개기·스쿠핑 태스크에서는 RDT-1B와 동급이거나 그 이상**이었다.

즉 §3.1의 압도적 우위는 "원 논문 저자의 기본 레시피 대 남의 기본 레시피" 비교이고, 파인튜닝 레시피를 통제하면 격차가 상당히 좁혀지거나 태스크에 따라 뒤집힌다 — VLA 벤치마크를 읽을 때 항상 따라붙는 주의사항이다.

## 4. 내 실습 연결

- **연속/discrete action 축의 반대쪽 끝**: 우리 [VLA 실습](../vla.md)에서 직접 돌린 [OpenVLA](openvla.md)는 7-DoF action을 256-bin으로 이산화해 "정책 학습 = 다음 토큰 예측"으로 만들었다. RDT는 이산화를 아예 하지 않고 diffusion denoising으로 연속 action chunk를 직접 생성한다. [VLA 리뷰 허브](vla.md)가 정리한 "연속 회귀 → 이산 분포"([DreamerV3](../world-models/dreamerv3.md)의 twohot, [MuZero](../world-models/muzero.md)의 categorical support, OpenVLA의 256-bin) 계열과 정반대 지점에 RDT를 놓을 수 있다 — action head 설계 스펙트럼(회귀 ↔ 이산 분류 ↔ diffusion)에서 diffusion 쪽 대표 사례로 참고할 값어치가 있다.
- **양팔은 우리 실습 범위 밖**: [매니퓰레이션](../manipulation.md)은 전부 단일 Franka 팔 + 고전 파이프라인(기하 인지 + analytic grasp + DLS IK)이고 학습 정책은 쓰지 않는다. RDT가 다루는 "두 팔 협응"은 우리 실습에 없는 완전히 다른 자유도 문제다. 다만 흥미로운 접점 하나: [매니퓰레이션](../manipulation.md)의 pouring 태스크가 정직하게 실패한 이유(얇은 컵 벽을 top-down 평행 그리퍼로 물고 130° 기울이다 파지가 미끄러져 컵째 낙하)는, 애초에 **한 팔로는 "쥐기"와 "안정화"를 동시에 못 하는 구조적 한계**였다. 두 번째 팔이 그릇이나 컵 몸통을 보조하는 양팔 협응이라면 이런 실패 모드 자체가 다른 문제로 바뀐다 — RDT류 양팔 파운데이션 모델이 겨냥하는 문제 유형을 우리 실패 사례로 구체화해볼 수 있는 지점이다.
- Physically Interpretable Unified Action Space(§2.3)는 "이종 로봇 데이터를 어떻게 한 모델에 섞을 것인가"라는, 우리가 겪지 않은(단일 로봇·단일 embodiment) 문제에 대한 해법이라 실습으로 검증할 기회는 없었다 — 순수 문헌 이해로 남겨둔다.

## 5. 한 줄 평 / 한계

**한 줄 평.** OpenVLA가 "VLM 인프라를 그대로 재사용"하는 실용주의였다면, RDT는 "diffusion의 표현력을 우선"하는 반대편 극단이다. 이산화 없이 다봉 action 분포를 직접 모델링하고, 물리적 의미를 보존하는 unified action space로 이종 로봇 사전학습의 데이터 희소성 문제를 정면 공략했다는 점이 핵심 기여이며, 가중치·코드·데이터셋 전부 공개해 양팔 VLA 연구의 공통 기준선 중 하나가 됐다.

**한계.**
- **수작업 action-space 매핑**: 통합 action 공간은 로봇마다 물리 슬롯을 사람이 정의·매핑해야 한다 — 완전히 자동화된 embodiment-agnostic 표현은 아니다.
- **자체 벤치마크 의존**: §3.1의 헤드라인 수치는 저자가 직접 수집한 ALOHA 태스크·기본 레시피 기준이다. LIBERO·SimplerEnv 같은 표준 공개 벤치마크에서의 폭넓은 비교는 원 논문에 상대적으로 적다(미확인: 후속 판(v2)에서 추가됐는지 여부).
- **후속 controlled 비교에서 격차 축소**: §3.2에서 보듯, 파인튜닝 레시피를 통일하면 OpenVLA-OFT·Diffusion Policy 대비 우위가 상당 부분 좁혀지거나 역전된다 — 원 논문의 벤치마크 수치를 곧이곧대로 "diffusion이 이산화보다 항상 낫다"로 일반화하기는 어렵다.
- **추론 비용**: 5-step으로 줄였다고는 해도 diffusion denoising은 구조적으로 OpenVLA류 단일 forward pass 토큰 예측보다 무겁다 — 381Hz라는 수치 자체는 RTX 4090 기준이라, 더 낮은 사양 하드웨어에서의 실사용성은 별도로 확인이 필요하다.
- **문헌 리뷰의 한계**: 이 리뷰는 원문·공식 저장소·프로젝트 페이지 인용에 기반했고 우리가 직접 재현·파인튜닝하지 않았다 — [OpenVLA](openvla.md) 리뷰처럼 "체크포인트를 돌려 논문값과 대조"하는 검증 단계가 빠져 있다는 점을 분명히 해둔다.
