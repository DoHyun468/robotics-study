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

## 내 실습 연결 — 내 하네스에서 직접 실행 (실측)

FT-LIBERO 체크포인트를 **내 WSL2 + RTX 4090 + 기존 LIBERO 하네스**(OpenVLA 평가에 쓴 그 체크아웃, 같은 bddl/init-state 파일)에서 실측했다. 프로토콜은 내 OpenVLA 런과 동일: **공식 고정 init state(에피소드 k → init_states[k]) + 10-step settle, 태스크당 2 트라이얼(suite당 20ep), suite별 step 예산 220/280/300/520**. RLDX는 서버-클라이언트(zmq)로 띄우고 action chunk 16 중 8-step 실행. 입력은 각 모델의 네이티브 규약(RLDX: front+wrist 2뷰+state / OpenVLA: 3인칭 1뷰) — 기술보고서의 baseline 비교와 같은 방식이다.

### 동일 조건 A/B + 확장 재현 런 (2026-08-11)

| Suite | OpenVLA-7B-FT (내 실측) | **RLDX-1 (내 실측, 20ep)** | **RLDX-1 (내 실측, 100ep)** | 보고서 주장 (500ep) |
|---|---|---|---|---|
| Spatial | 80% | 95% | **100.0%** | 98.0 |
| Object | 85% | 100% | **99.0%** | 99.3 |
| Goal | 85% | 90% | **93.0%** | 98.4 |
| Long (libero_10) | 45% | 95% | **96.0%** | 95.3 |
| **평균** | 73.75% | 95.0% | **97.0%** (388/400) | 97.8 |

(20ep 런 = OpenVLA와 완전 동일 프로토콜의 A/B, 100ep 런 = 같은 고정-init 하네스로 표본만 5배 키운 재현 런.)

읽는 법:
- **주장이 내 셋업에서 재현된다**: 400ep 표본에서 **97.0 vs 주장 97.8** — Spatial/Object/Long은 오차 안에서 일치(100.0/99.0/96.0 vs 98.0/99.3/95.3). 유일하게 Goal(93.0 vs 98.4)이 5%p 낮은데, 실패가 특정 태스크에 몰려 있어(top-drawer+bowl, cream-cheese-in-bowl 등 4개) 프로토콜 차이(고정 init vs 랜덤 리셋, step 예산 300 vs 720)가 원인 후보다.
- **격차의 위치가 논문 서사와 일치**: 짧은 suite에서 OpenVLA 대비 +10~15%p, **Long에서 +50%p** — "long-horizon으로 갈수록 벌어진다"(RC365 결과의 축소판)를 내 하네스에서 재확인.
- **실패 4건은 전부 부분 실패(1/2)**: spatial `black_bowl_on_the_stove`(내 OpenVLA도 stove 계열에서 실패 — 겹치는 실패 모드), goal `top_drawer+bowl`·`cream_cheese_in_bowl`, long `mug_in_microwave_and_close`.
- 결과 원본: `robotics-lab/outputs/rldx_ab_n2.json`(태스크별), 롤아웃 영상 80개 저장.

### 재현 과정에서 확인한 것 (정직 기록)

### 재현 과정에서 확인한 것

- 공개 상태는 진짜다: 코드·가중치·문서(architecture/training/evaluation.md)까지 전부 실재. `RLDXPolicy` 5줄로 로드된다.
- **flash-attn 벽**: 표준 설치 경로가 CUDA toolkit(nvcc) 전제의 소스 빌드. nvcc 없는 WSL에서는 커뮤니티 프리빌트 휠(`2.7.4.post1+cu126torch2.7`)로 우회해야 했다.
- **`setup_libero.sh`는 그대로는 안 돈다**: ① cmake 4.x가 구식 egl-probe를 거부(`CMAKE_POLICY_VERSION_MINIMUM=3.5` 필요) ② 스크립트가 고정한 transformers 4.51.3에는 `masking_utils`가 없어 자기 자신(rldx 패키지)을 import 못 한다(→4.57.0으로 상향) ③ diffusers/accelerate가 클라이언트 venv에 누락 ④ 무핀 mujoco가 3.11로 풀려 robosuite 1.4의 `MjData.qM` 접근이 깨진다(→내 검증 조합 mujoco 2.3.2로 고정). — 대기업 공개 레포도 환경 재현은 이만큼 깨지기 쉽다는 표본.
- **평가 프로토콜 차이 발견**: 그들의 LIBERO 래퍼는 에피소드마다 **랜덤 초기 배치**로 리셋한다(공식 LIBERO/OpenVLA 프로토콜은 벤치마크 고정 init state). A/B 공정성을 위해 고정-init 옵션을 패치로 추가했다(기본 동작 불변, `RLDX_FIXED_INIT=1`).

## 한 줄 평 / 한계

- **강점**: "기능 축" 문제 정의가 선명하고, 모듈 하나하나가 실기 태스크 하나씩과 짝지어 실증된다(컨베이어↔모션, 셸게임↔메모리, 계란↔촉각). MM-DiT의 스트림 분리를 modality 통합에 재활용한 MSAT는 "없는 신호는 마스킹으로 끈다"는 점에서 이질적 로봇 데이터 현실에 잘 맞는 설계.
- **캐비엇**: ① 비교표는 전부 자체 평가(baselines도 자기들이 파인튜닝) ② 파라미터 표기 혼재 — HF 카드 기준 PT 6.9B/MT 8.1B를 1차 출처로 ③ 가중치는 비상업 라이선스(연구·평가·리뷰는 가능, 이 페이지도 그 용도) ④ 실기 태스크는 자사 하드웨어(ALLEX) 중심이라 외부 재현이 어렵다 — 시뮬 재현(LIBERO)이 외부인이 만질 수 있는 유일한 접점이고, 그래서 내 실측이 여기 붙는다.
- **내 트랙과의 접점**: 데이터 엔진 축 — 그들의 "사람 손 리타게팅 시간당 200+ 시연"과 LeRobot v2.1 스키마는 내 human demo→VLA 파이프라인(HM 트랙 + vla_datapipe)이 겨냥하는 바로 그 지점이다. DexBench의 Contact Precision 축은 내 HM5 force-closure(ε)·침투 페널티 실습과 같은 언어를 쓴다.
