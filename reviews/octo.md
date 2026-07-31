# Octo (2024)

*오픈 제너럴리스트 로봇 정책 — transformer + diffusion action head*

Octo Model Team (Ghosh, Walke, Pertsch 외), *Octo: An Open-Source Generalist Robot Policy*, RSS 2024 (arXiv 2405.12213). 카테고리 개요는 [VLA 리뷰 허브](vla.md), 실습 기록 전체는 [VLA 실습 페이지](../vla.md). [OpenVLA 리뷰](openvla.md)와 짝을 이루는 비교 대상.

## 한 줄 요약

> Open X-Embodiment 25개 데이터셋·80만 궤적으로 사전학습한 **소형(27M/93M) transformer 정책**. 언어 지시문·목표 이미지·이미지 히스토리를 유연하게 받는 "readout 토큰" 설계와, action을 회귀가 아니라 **diffusion으로 디코딩**하는 head가 특징이다. 가중치·코드·학습 파이프라인이 전부 공개(JAX/Flax)됐고, LIBERO에서 93M 모델이 7B OpenVLA와 거의 같은 성적을 낸다 — "스케일이 전부는 아니다"를 보여주는 실증 사례.

---

## 1. 문제 — "제너럴리스트 정책"을 누구나 미세조정할 수 있게

OpenVLA와 문제의식은 겹치되 강조점이 다르다. Octo 저자들은 로봇 파운데이션 모델이 실제로 쓸모 있으려면 다음이 필요하다고 본다.

1. **이기종(heterogeneous) 로봇·센서에 대한 유연성**: 실험실마다 카메라 개수·배치, 로봇 팔·그리퍼, action 공간(엔드이펙터 delta vs 조인트)이 다 다르다. 사전학습 때 본 적 없는 조합이라도 정책 백본을 다시 학습하지 않고 새 입출력 헤드만 붙여 쓸 수 있어야 한다.
2. **가벼운 미세조정**: 새 로봇·태스크에 적응하는 데 대형 클러스터가 필요하면 "오픈"의 의미가 퇴색한다. 소비자급 GPU, 수 시간, 수십~백여 개 시연이면 충분해야 한다.
3. **다중 모달 action 분포 표현**: 시연 데이터는 같은 상황에서도 사람마다 다른 방식으로 태스크를 수행한 궤적이 섞여 있다(multi-modal). 단순 회귀(MSE) 손실은 이런 분포를 평균 내버려 어색한 행동을 만든다.

RT-1/RT-2 계열과 OpenVLA는 대부분 고정된 관측·action 포맷을 가정한다. Octo는 "사전학습된 백본은 그대로 두고, 관측·action 포맷만 갈아 끼우는" 아키텍처로 이 문제에 접근한다.

## 2. 방법

### 2.1 아키텍처 — 토큰화 + 블록형 causal transformer + readout 토큰

$$
\text{관측 } o_{1:t},\ \text{태스크(언어/목표이미지)} \;\longrightarrow\; \text{토큰화} \;\longrightarrow\; \text{Transformer} \;\longrightarrow\; \text{readout 토큰} \;\longrightarrow\; \text{action head}
$$

- **입력 토큰화**: 이미지는 얕은 CNN으로 패치화(3인칭 카메라 256토큰, 손목 카메라 64토큰), 언어 지시문은 사전학습된 **t5-base**로 인코딩해 16개 임베딩 토큰으로 투입.
- **블록형 마스킹 attention**: 관측 토큰은 같거나 이전 타임스텝의 토큰에만 causal하게 주의를 준다 — 시계열 구조를 명시적으로 반영.
- **readout 토큰**: 학습되는 특수 토큰으로, 자신보다 앞선 관측·태스크 토큰에는 주의를 주지만 **어떤 관측·태스크 토큰도 readout 토큰에는 주의를 주지 않는다**. 즉 readout 토큰은 정보를 "수동적으로 읽어가기만" 할 뿐 본체 표현에 영향을 주지 않는다 — 이 비대칭 설계 덕분에 **미세조정 시 새 관측 입력이나 새 action head를 자유롭게 붙이거나 뗄 수 있다**(핵심 transformer 가중치 재학습 불필요).
- **모델 크기**: **Octo-Small(27M, 12층, hidden 384)**, **Octo-Base(93M, 12층, hidden 768)** 두 버전.

### 2.2 Diffusion action head — 회귀 대신 디노이징

OpenVLA가 action을 이산 토큰으로 만들어 "언어모델처럼" 예측한다면, Octo는 정반대 방향을 택한다: readout 토큰을 조건으로 **conditional diffusion**으로 action을 생성한다.

$$
a \sim p_\theta(a \mid z_{\text{readout}}),\quad \text{가우시안 노이즈에서 시작해 } K\text{-step 디노이징}
$$

- 3-layer MLP(hidden 256) 디노이징 네트워크, **20-step** 코사인 노이즈 스케줄.
- 단일 timestep이 아니라 **action chunk**(짧은 구간의 연속 action)를 한 번에 예측 — action chunking으로 실행 안정성을 높이는 설계는 Diffusion Policy 계열과 궤를 같이한다.
- 회귀(MSE) 대신 diffusion을 쓰는 이유가 바로 §1의 "다중 모달 action 분포" 문제: 디노이징 과정은 여러 모드를 가진 분포도 표현할 수 있어, 시연자마다 다른 궤적을 뭉개지 않는다.

### 2.3 학습

- **데이터**: Open X-Embodiment 중 **25개 데이터셋을 큐레이션한 약 80만 궤적**, 다양한 로봇 embodiment·씬·태스크. 데이터셋 간 다양성을 반영해 수동으로 샘플링 가중치를 조정.
- **컴퓨트**: Octo-Base 기준 **TPU v4-128 pod에서 30만 step, batch size 2048, 약 14시간** 학습. lr 3e-4, inverse-square-root decay, 2000-step warmup.
- 학습 신호는 OpenVLA와 마찬가지로 **순수 모방학습**(행동 복제) — 보상 신호 없음.

### 2.4 미세조정 — 새 embodiment에 저비용으로 적응

- 표준 레시피: 타겟 도메인 시연 **약 100개**, 5만 미세조정 step, 코사인 decay.
- **단일 NVIDIA A5000(24GB) GPU에서 약 5시간**이면 새 로봇·카메라 구성·action 공간에 적응 — 새 관측 입력이나 action head를 추가해도 사전학습된 트랜스포머 본체는 그대로 재사용(§2.1의 readout 토큰 설계 덕).
- 코드·가중치·학습 파이프라인 전부 공개(JAX/Flax, `github.com/octo-models/octo`) — OpenVLA(PyTorch)와 함께 오픈 VLA 생태계의 양대 기준선.

## 3. 결과

### 3.1 원문 결과 (Octo 논문, WidowX/UR5/RT-1 로봇 등 9개 플랫폼)

- **제로샷**(사전학습 체크포인트 그대로): WidowX 0.50, UR5 0.70, RT-1 로봇 0.80 (성공률).
- **미세조정 6개 평가 과제 평균**: 처음부터 학습한 baseline 대비 **52% 우위**(원문 claim).
- 새 관측(추가 카메라)·새 action 공간(조인트 제어 등)으로도 미세조정 성공 — "제너럴리스트 백본 + 갈아 끼우는 헤드" 설계가 실제로 작동함을 보임.

### 3.2 LIBERO 벤치마크 — Octo(93M) vs OpenVLA(7B)

Octo 원 논문에는 LIBERO 결과가 없지만, 이후 OpenVLA 계열 논문들(OpenVLA 및 OpenVLA-OFT)이 Octo를 LIBERO에 미세조정해 baseline으로 보고했다.

| suite | Octo (93M) | OpenVLA (7B) |
|-------|-----------|--------------|
| spatial | 78.9% | 84.7% |
| object | 85.7% | 88.4% |
| goal | 84.6% | 79.2% |
| long | 51.1% | 53.7% |
| **평균** | **75.1%** | **76.5%** |

파라미터 수로는 **OpenVLA가 Octo의 약 75배(7B/93M)**인데, LIBERO 평균 성공률 차이는 **1.4%p**에 불과하다. 우리가 직접 재현한 OpenVLA 실측 평균(74%, [OpenVLA 리뷰](openvla.md) §3.2)과 비교해도 Octo(75.1%, 원문 수치)가 오히려 근소 우위다.

## 4. 내 실습 연결

- 우리가 [OpenVLA를 LIBERO 4-suite에서 직접 재현](openvla.md)했을 때(평균 74%) 이미 이 지점을 언급했다: **"LIBERO에선 7B의 20배 파라미터가 거의 안 드러난다."** Octo 원문 수치(93M, 75.1%)까지 나란히 놓고 보면 그 관찰이 더 뚜렷해진다 — LIBERO처럼 사전학습 분포와 평가 분포가 비교적 가까운 벤치마크에서는, **대형 VLM 백본의 일반화 이점이 잘 드러나지 않고 오히려 diffusion 기반 소형 정책이 동급 이상**을 낸다. 스케일이 항상 답은 아니라는 근거가 하나 더 늘어난 셈이다.
- 우리 [컨텍스트 사다리](../context.md) L3(언어+이미지→action) 안에서도 Octo는 OpenVLA와 서로 다른 "출력단 철학"을 보여주는 대조군이다: OpenVLA는 action을 **이산 토큰**(256-bin, 언어모델링과 동일 손실)으로, Octo는 **diffusion 디노이징**(연속·다중모달 분포)으로 만든다. [DreamerV3](../world-models/dreamerv3.md)의 twohot·[MuZero](../world-models/muzero.md)의 categorical support가 "연속→이산" 계열이라면, Octo는 정반대로 "연속 분포를 직접 모델링"하는 계열 — 같은 문제(정책의 action 출력)에 대한 두 갈래 해법이 VLA 안에서도 공존한다.
- **다음 실습 후보로 유력하다.** OpenVLA는 우리가 이미 재현·파인튜닝까지 끝낸 상태고, Octo는 (1) 파라미터가 20~75배 작아 우리 GPU 예산으로 **풀 파인튜닝(LoRA 아닌 전체 학습)까지 현실적**이고, (2) diffusion head라는 다른 action 표현 방식을 직접 만져볼 수 있고, (3) JAX/Flax 스택이라 우리 환경(WSL2, PyTorch 중심)과는 별도로 새로운 환경 셋업 경험(JAX+TPU/GPU, `octo-models/octo` 레포)이 쌓인다는 점에서 매력적이다. LIBERO에 우리 손으로 미세조정해 위 표의 75.1%를 재현해보면, OpenVLA 재현 때와 마찬가지로 "논문 수치 재현 vs 직접 학습시켜 그 수치에 도달하기" 사이의 간극을 다시 확인할 수 있을 것이다.

## 5. 한 줄 평 / 한계

**한 줄 평.** OpenVLA가 "VLM을 로봇 정책으로 억지로 얹는" 접근이라면, Octo는 "로봇 정책을 처음부터 유연하게 설계"한 접근이다. 두 철학이 LIBERO에서 거의 같은 성적을 낸다는 사실 자체가 이 분야에 던지는 질문이 크다 — **지금 VLA 성능을 가르는 건 백본 스케일보다 데이터·태스크 분포와의 정합성일 수 있다.**

**한계.**
- **LIBERO는 원 논문의 벤치마크가 아니다**: 위 §3.2 수치는 후속 논문(OpenVLA-OFT 등)이 재현·보고한 것으로, Octo 저자들이 직접 튜닝한 세팅과는 다를 수 있다 — 두 모델의 "공정 비교"로 못 박기엔 조심스럽다(미확인 여지 있음).
- **diffusion head의 추론 비용**: 20-step 디노이징은 자기회귀 토큰 예측보다 스텝당 계산량이 더 들 수 있다 — 고주파 제어에서 실질 지연시간이 얼마나 되는지는 우리가 직접 벤치마크해봐야 확인 가능(미확인).
- **언어 이해 깊이**: t5-base 기반 지시문 인코딩은 Llama-2 7B급 언어 백본에 비해 복잡한 지시문·추론이 필요한 태스크에서 약할 가능성이 있다 — OpenVLA류 대형 VLM 백본이 여전히 유리한 영역(복잡한 지시 따르기, 새로운 물체 개념)이 남아있을 것으로 보인다.
- **9개 플랫폼 제로샷 수치의 일반화 폭**: 원문 claim(52% 우위 등)은 저자 자체 평가 세팅 기준이라 외부 재현이 필요하다 — 우리가 다음 실습에서 직접 확인해볼 지점.
