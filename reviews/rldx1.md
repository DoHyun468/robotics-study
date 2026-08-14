# RLDX-1 — RLWRLD의 Dexterity-First 로봇 파운데이션 모델

*기술보고서를 직접 읽고, 공개 코드·가중치를 내 4090 + 내 LIBERO 하네스에서 실제로 돌려보며 정리*

RLWRLD, *RLDX-1 Technical Report*, arXiv:2605.03269 (v2, 2026-05). 저자 68명(lead Dongyoung Kim, KAIST 겸직 다수). [코드](https://github.com/RLWRLD/RLDX-1)(Apache-2.0) · [가중치](https://huggingface.co/RLWRLD)(RLWRLD Model License v1.0, **비상업**) · [프로젝트](https://www.rlwrld.ai/en/rldx-1) · [DexBench](https://dexbench.org/en/)

## 한 줄 요약

> 표준 VLA(π0.5, GR00T)가 못 하는 세 가지 — **움직임 인지(motion awareness), 장기 기억(long-term memory), 물리 감각(tactile/torque)** — 를 각각 전용 모듈(STSS 모션 모듈, cognition-feature FIFO 메모리, physics 스트림)로 붙이고, 이 이질적 modality들을 **MSAT**(MM-DiT를 action으로 확장한 multi-stream joint-attention transformer)로 한 번에 디노이즈하는 7~8B VLA. LIBERO 97.8 등 시뮬 8종 전승 주장 + ALLEX 실기 86.8 vs 40%대(π0.5/GR00T)로 "기능 축" 우위를 실증했다.

## 문제 설정

VLM에서 물려받은 versatile intelligence(장면 이해·언어 일반화)만으로는 실세계 조작이 안 되는 지점이 있다:

1. **컨베이어 위 물체 집기** — 정지 관측으로는 다음 위치를 못 맞춘다 (motion)
2. **셸 게임** — 현재 프레임에는 정답 정보가 없다 (memory)
3. **플러그 삽입·계란 집기** — 시각 변화가 거의 없고 가려진다 (physics)

기존 VLA는 단일 프레임 + 현재 관측 + 시각 전용이 기본이라 세 축 모두 구조적으로 막혀 있다.

## 아키텍처

### 입출력

$$(c_t,\ o_{t-K:t},\ s_t,\ p_t)\ \longmapsto\ a_{t:t+H}$$

언어 지시 $c_t$, $K{+}1$프레임 비디오 $o_{t-K:t}$(기본 4프레임, 상대 오프셋 $\{-6,-4,-2,0\}$), proprioception $s_t$, 물리 신호 $p_t$(토크/촉각, 옵션) → 길이 $H{+}1$ action chunk(FR3 16 / ALLEX 40).

### RLDX-1-VLM (Qwen3-VL 8B 개조)

- **robot-VQA 파인튜닝**: 로봇 궤적에서 (1) EE↔물체 공간관계 (2) 중간 subtask (3) 현재 프레임의 저수준 action, 3종 VQA를 만들어 Qwen3-VL을 적응 → RoboCasa 57.5→60.9%.
- **중간층 추출**: action 디코딩에 최종층이 아닌 **Layer 18** hidden state 사용(상위층은 언어 생성 특화). Ablation이 극적: L8→51.1, **L18→60.9**, L28→56.3.
- **Cognition tokens**: 학습형 query 토큰 **64개**를 입력 뒤에 붙여 $x=[v_t, l_t, q]$로 통과시키고 $q$ 위치 출력만 cognition feature $h_t$로 사용 — visual+언어 문맥을 action에 필요한 만큼만 증류.
- 학습 시 LLM 백본은 **상위 4층만** 풀고 나머지 동결.

### Functionality 1 — Motion Module (STSS)

비전 인코더(27층 ViT) **9층 뒤**(≈30% 깊이, "물리 단서가 가장 풍부한 깊이")에 residual로 삽입:

$$\tilde v^{(i)}_t = v^{(i)}_t + S_\theta(\mathrm{STSS}(v^{(i)}_t))$$

STSS(space-time self-similarity)는 각 시공간 feature와 이웃의 상관 텐서 — optical-flow류의 명시적 움직임 표현을 인코더 내부에 심는다. LLM 쪽은 앞 4층(Qwen3-VL DeepStack 구간)까지 멀티프레임을 그대로 통과시킨 뒤 **과거 프레임을 average-pool 한 토큰으로 압축**해 나머지 층을 통과 — 연산은 아끼고 시간 문맥은 유지.

### Functionality 2 — Memory Module

VLM 직후, action-chunk 경계마다 cognition feature를 FIFO 캐시:

$$Q_t=[h_{t-n H'},\dots,h_{t-2H'},h_{t-H'}],\quad H'=H{+}1,\ n_{\text{mem}}=3$$
$$m_t = M_\theta([Q_t, h_t]) \quad (\text{causal attention})$$

시간 창 = $n_{\text{mem}}\times(H{+}1)$ = **FR3 48 / ALLEX 120 스텝**. $h_t$와 $m_t$를 둘 다 action model에 넘긴다. 셸 게임(91.7% vs 랜덤 수준 baseline)이 이 모듈의 존재 증명.

### Action Model — flow-matching DiT + MSAT

Flow matching 표준형: $\tau\in[0,1]$, $\epsilon\sim\mathcal N(0,I)$, $a^\tau = \tau a + (1-\tau)\epsilon$에 대해

$$\mathcal L(\theta)=\big\|u_\theta(a^\tau_{t:t+H},\tau,c_t)-(a_{t:t+H}-\epsilon)\big\|_2^2,\qquad c_t=[h_t,m_t,s_t,p_t]$$

추론은 Euler $T$스텝: $a^{\tau_{i+1}}=a^{\tau_i}+(\tau_{i+1}-\tau_i)\,u_\theta(a^{\tau_i},\tau_i,c_t)$.

**MSAT** = FLUX/SD3의 MM-DiT(double→single stream)를 action에 이식:
- 초기 **double-stream**: C 스트림 $[h_t, m_t]$ ‖ A 스트림 $[s_t, a^\tau]$ — 스트림별 norm/QKV를 따로 두고, Q·K·V를 토큰 축으로 이어붙여 **joint self-attention** 후 다시 분리·residual.
- 후기 **single-stream**: C·A를 한 시퀀스로 병합.
- 물리 신호가 있으면 **P 스트림 추가** → triple-stream/double-stream으로 확장. 신호가 없는 embodiment에서는 P 스트림 attention을 마스킹으로 끔 — "있을 때만 켜는" 모듈러 설계.
- 세부: A 스트림에 RoPE(chunk 내 상대시간), $\tau$는 adaLN이 아니라 **in-context 토큰**으로 주입, RMSNorm(+QK-norm)·SwiGLU. embodiment별 I/O projection + 배치 일부를 **공유 embodiment-agnostic projection**으로 통과시켜 새 로봇 적응용 초기화를 확보.

### Functionality 3 — Physics Stream

P 스트림은 $p_t$를 따로 처리하며 보조 목표로 **미래 물리 신호 자체를 flow-matching으로 예측**:

$$p^\tau_{t+1:t+L}=\tau p_{t+1:t+L}+(1-\tau)\epsilon_p,\qquad L=H{+}1$$

action과 물리 신호를 **동시에 디노이즈** — 접촉 동역학을 내재화시키는 장치. mid-training에서 P 스트림 출력 가중치는 **near-zero 초기화** + 새 modality 입력에 dropout 0.3 + 처음 2K step은 기존 파라미터 동결(alignment warmup). 효과: Plug Insertion 33.3 vs 16.7/20.8, Egg PnP 61.1 vs 37.5/45.8.

## 데이터와 학습

**사전학습 1.5M 에피소드**: OXE 870K + DROID 92K + Galaxea 114K + AgiBot World(G 239K/H 36K) + Fourier ActionNet 30K + Humanoid Everyday 9K + **합성 150K**. 프레임당 vision 토큰 ≤64(native-resolution). state/action은 per-dataset **q01/q99로 [-1,1] 정규화**(OpenVLA와 같은 관례). 100K step, gbs 8192, **64×H200 195시간**.

**합성 데이터 파이프라인**(GR-1·ALLEX 증폭): 소스 시연의 초기 프레임+지시로 I2V 생성 → **IDM으로 action 라벨** → (1) VLM 기반 태스크 증강(behavior/object/placement/hand 4-요소 분해 재조합 + skill-primitive 조건 변형) (2) 장면 증강(FLUX.2-dev I2I + Canny, Cosmos-Transfer2.5-2B V2V) → 2단 필터: VLM 품질 평가(지시 부합·궤적 그럴듯함 1–5점) + **motion-consistency 필터**(IDM action을 시뮬 재생한 rollout과 생성 비디오를 frozen V-JEPA2 + cross-attention probe로 정합 판정). 효과: GR-1 합성 5× 스케일업, ALLEX Pot-to-Cup grasp 66.7→**83.3%**.

**mid-training**(25K step, 15시간): ALLEX(자체 teleop+합성 5:5), FR3(DROID 8 : 자체 촉각/토크 2) — 기능 3축이 여기서 켜진다.

**post-training**: 기본 IL + **adaptive data collection**(일관성 요소/변동 요소를 나눠 기본 수집 → 실패 조건 타깃 보강 수집 반복) + **RL(RECAP 기반)**: VLM critic이 **네이티브 숫자 토큰으로 가치를 텍스트 예측**(새 head 없음 → 적은 데이터로 전이) → advantage-conditioned supervision.

## 추론 최적화

로봇 제어는 지연이 곧 성능(관측→실행 사이 장면이 변한다):
1. **Static graph 변환**: RoPE·attention mask 등 설정 의존 연산을 forward 밖으로 빼 **CUDA Graph 하나**로 캡처(Torch Compile의 subgraph 분절 제거).
2. **수제 fused kernel**: short-prefill 워크로드에서 RMSNorm+RoPE+Attention을 한 커널로(중간 Q/K의 글로벌 메모리 왕복 제거), Nsight로 병목 그룹 프로파일 후 선택 교체.

결과: **43.7ms/step @ RTX 5090 (1.63×, >22Hz)**. (내 4090에서는 이 경로가 부분 지원 — 아래 실측.)

## 결과 (보고서 주장 — RLWRLD 자체 평가)

시뮬(각 벤치 파인튜닝 후):

| | LIBERO | L-Plus | G-VM | G-VA | WidowX | RC-Kitchen | GR-1 | RC365 |
|---|---|---|---|---|---|---|---|---|
| π0.5 | 96.9 | 86.5 | 72.7 | 68.4 | 46.9 | 62.1 | 15.4 | 16.9 |
| GR00T N1.6 | 96.7 | 72.6 | 76.1 | 57.1 | 57.1 | 66.2 | 47.6 | 26.9 |
| **RLDX-1** | **97.8** | **86.7** | **81.5** | **77.4** | **71.9** | **70.6** | **58.7** | **32.1** |

패턴: 쉬운 벤치일수록 격차가 좁고(LIBERO +0.9p), **강건성 변형(GVA +9p)과 long-horizon 조합(RC365 Composite 19.0 vs 12.6)에서 벌어진다**. GR00T N1.6은 RoboCasa 시뮬 데이터로 사전학습하고도 RC-Kitchen에서 밀렸다(시뮬 사전학습 0인 RLDX에).

실기 3플랫폼: OpenArm(범용성 6태스크 전승, Object Grounding 87.5 vs GR00T 33.3=랜덤), **ALLEX 4태스크 평균 86.8 vs 39.1/44.8**(motion/memory/physics 각각이 해당 모듈의 ablation 실증 구도), FR3 6태스크 평균 68.5 vs 34.4/31.6.

## 내 실습 연결

**내 4090 + 내 LIBERO 하네스에서 직접 실측했다** — OpenVLA 동일조건 A/B(20ep/suite)에서 **95.0 vs 73.75%**(Long +50%p), 고정-init 하네스 400ep 재현에서 **97.0 vs 주장 97.8**. 프로토콜(고정 init state 정렬 패치), suite·태스크별 수치, 재현 실록(flash-attn/setup_libero 4중 수리/프로토콜 차이 발견), 그리고 내 human 시연을 LeRobot v2.1로 직렬화해 **RLDX 자체 로더로 검증(LOADER_OK)** 한 데이터 파이프라인까지 — 실전 전체는 전용 페이지 **[RLDX-1 실전](../rldx.md)** 에 정리했다.

## 한 줄 평 / 한계

- **강점**: "기능 축" 문제 정의가 선명하고, 모듈 하나하나가 실기 태스크 하나씩과 짝지어 실증된다(컨베이어↔모션, 셸게임↔메모리, 계란↔촉각). MM-DiT의 스트림 분리를 modality 통합에 재활용한 MSAT는 "없는 신호는 마스킹으로 끈다"는 점에서 이질적 로봇 데이터 현실에 잘 맞는 설계.
- **캐비엇**: ① 비교표는 전부 자체 평가(baselines도 자기들이 파인튜닝) ② 파라미터 표기 혼재 — HF 카드 기준 PT 6.9B/MT 8.1B를 1차 출처로 ③ 가중치는 비상업 라이선스(연구·평가·리뷰는 가능, 이 페이지도 그 용도) ④ 실기 태스크는 자사 하드웨어(ALLEX) 중심이라 외부 재현이 어렵다 — 시뮬 재현(LIBERO)이 외부인이 만질 수 있는 유일한 접점이고, 그래서 내 실측이 여기 붙는다.
- **내 트랙과의 접점**: 데이터 엔진 축 — 그들의 "사람 손 리타게팅 시간당 200+ 시연"과 LeRobot v2.1 스키마는 내 human demo→VLA 파이프라인(HM 트랙 + vla_datapipe)이 겨냥하는 바로 그 지점이다. DexBench의 Contact Precision 축은 내 HM5 force-closure(ε)·침투 페널티 실습과 같은 언어를 쓴다.
- **계열 리뷰**: 데이터 엔진(다지손 시연 수집)의 학계 대응물은 [DexCap](dexcap.md), physical sensing 스트림(촉각 융합)의 대응물은 [3D-ViTac](3d-vitac.md) — RLDX가 "토큰 스트림"으로 넣는 촉각을 3D-ViTac은 "공유 좌표계의 점"으로 넣는다는 문법 대비가 볼거리다.
