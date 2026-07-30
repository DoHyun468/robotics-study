# VLA — OpenVLA (LIBERO 재현 + 파인튜닝)

컨텍스트 전달 사다리의 최상단([context.md](context.md) L1 포인터 → L2 언어 grounding → **L3 VLA**). L1/L2는 "인지 → 우리 IK/제어"였지만, VLA는 **언어+이미지에서 action을 직접** 뱉는다 — 손으로 쌓은 인지·결정·IK·grasp([manipulation.md](manipulation.md), [grasp_sota.md](grasp_sota.md)) 전체를 정책 하나가 대체하는 셈. VLA는 커스텀 씬이 아니라 **자기 네이티브 벤치(LIBERO)**에서 평가해야 embodiment/action-space가 맞아 의미가 있으므로, 그대로 재현했다.

## 무엇인가 — OpenVLA-7B

**OpenVLA-7B**(RSS'24)는 Prismatic VLM(**Llama-2 7B + DINOv2 & SigLIP** 듀얼 비전 인코더) 위에 로봇 action을 **토큰으로** 학습시킨 오픈 VLA다. Open X-Embodiment(~1M 궤적)로 대규모 사전학습한 뒤, downstream 태스크는 **LoRA**로 가볍게 파인튜닝한다.

- **입력**: 3인칭 RGB(256²) + 자연어 지시
- **출력**: next-token 예측으로 뽑는 **7-DoF delta EE action**(+ gripper) — 연속 action을 discrete token으로 매핑해 BC(behavior cloning)로 학습
- 재현에는 LIBERO 4-suite에 **LoRA(r=32)**로 파인튜닝된 공식 체크포인트를 그대로 사용(`openvla/openvla-7b-finetuned-libero-spatial`, ~14GB 자동 다운로드)

## LIBERO 벤치마크 & 평가 프로토콜

LIBERO는 로봇 조작 언어-조건부 정책의 표준 벤치로, **spatial / object / goal / long(10)** 4개 suite로 난이도·조합축을 나눈다. 태스크는 전부 "pick up the black bowl {between the plate and the ramekin / on the cookie box / in the top drawer / ...} and place it on the plate" 류 — **언어로 지정된 물체를 집어 목표 위치에 놓기**다.

- 평가 단위: **20 에피소드/suite**(10 tasks × 2 trials) — 논문 full-eval(500-ep)의 서브셋
- `center_crop` 전처리 적용(학습 시 randomize 대응)
- 시뮬레이터: **robosuite 1.4.1 + mujoco 2.3.2** (LIBERO가 고정하는 조합)
- 렌더: `MUJOCO_GL=egl`(WSL2 EGL 오프스크린)

## 환경 / 의존성 — WSL2 dep-hell 실기록

전용 conda env `ov`(기존 `zg` env를 복제해 torch2.2+cuda+gcc 툴체인 재사용). 실제로 넘은 블로커들:

- **flash-attn 2.5.5**: py3.11 조합이라 소스 빌드 대신 **cp311 prebuilt wheel(cu122/torch2.2)**을 그대로 사용해 컴파일을 회피.
- **robosuite 1.4.1**: `pynput → evdev` 네이티브 빌드가 실패 → `--no-deps`로 설치 후 **런타임에 필요한 의존성만** 채움(teleop 미사용이라 evdev 자체가 불필요). robosuite 호환 버전으로 **mujoco 2.3.2** 고정.
- **LIBERO**: `pip install -e .` 후 `~/.libero/config.yaml`을 **미리 생성**해둠 — 첫 import 시 뜨는 `input()` 프롬프트가 비대화형 셸에서 `EOFError`로 죽는 걸 preseed로 회피.
- **protobuf 삼각충돌**: `tensorflow 2.15`(protobuf `<5` 요구) · `tensorflow-metadata`(최신 1.21은 `protobuf>=5.26` 요구) · `wandb`(`protobuf 4.x` 요구)가 서로 맞물려 어느 조합도 동시 만족이 안 됨 → **tensorflow-metadata 1.14 + protobuf 4.25.3 + tfds 4.9.3**로 수렴시켜 해소.
- 종료 시 `MUJOCO_GL=egl` cleanup에서 뜨는 EGL `__del__` 관련 로그는 무해한 노이즈(`EVAL_RC=0` 확인).

## 재현 결과 — LIBERO 4-suite

| suite | 내 재현(20-ep) | 논문(500-ep) | Octo 논문 |
|---|---|---|---|
| spatial | 80% | 84.7 | 79 |
| object | 85% | 88.4 | 86 |
| goal | 85% | 79.2 | 85 |
| long(10) | 45% | 53.7 | 51 |
| **평균** | **74%** | 76.5 | 75.1 |

spatial suite 기준으로는 20 에피소드 중 **16/20 = 80% 성공**, 실패 4개는 table center·stove 등 애매한 위치에서 grasp를 놓친 케이스. 4-suite 전체로 보면 논문과 **패턴이 정합**한다(spatial~goal은 높고, long이 최난이도인 순서가 동일). 흥미로운 지점은 **7B짜리 OpenVLA가 93M짜리 Octo를 평균 ~1.4%p밖에 못 앞선다**는 것 — LIBERO라는 벤치 안에서는 파라미터 20배 차이가 성능에 거의 반영되지 않는다. 즉 **스케일이 항상 답은 아니다.** 80%가 논문값과 정합한다는 사실 자체가, 재현·통합(모델 로드, LIBERO 인터페이스, action 디코딩, 시뮬 루프)이 올바르게 됐다는 검증이기도 하다.

## 실제 롤아웃

<div style="display:flex;gap:12px;flex-wrap:wrap"><video src="_static/vla_ok1.mp4" controls loop muted playsinline style="width:31%;min-width:220px;border-radius:8px"></video><video src="_static/vla_ok2.mp4" controls loop muted playsinline style="width:31%;min-width:220px;border-radius:8px"></video><video src="_static/vla_fail.mp4" controls loop muted playsinline style="width:31%;min-width:220px;border-radius:8px"></video></div>

*"pick up the black bowl … place it on the plate" — 성공 2 / 실패 1*

20개 롤아웃 전부 성공/실패 라벨을 붙여 저장했고, 그중 대표 3개(성공 2 + 실패 1)를 위에 임베드했다.

## 자체 LoRA 파인튜닝 — 파이프라인은 동작, 단 짧은 run은 undertrain

평가만 재현하는 걸 넘어, 단일 4090에서 `openvla-7b` 베이스 위에 **LoRA 파인튜닝을 처음부터 끝까지 직접 돌렸다**: RLDS 데이터 변환 → LoRA 학습(`finetune.py`) → 어댑터 merge → LIBERO eval — **파이프라인 자체는 end-to-end로 정상 구동**한다.

문제는 학습량이다. GPU 시간 제약으로 돌린 bounded run은 **1500 step ≈ 0.45 epoch**밖에 안 되고, 이 체크포인트로 LIBERO-Spatial을 평가하면 **0%**가 나온다. 여기서 중요한 건 원인 규명이다 — action normalization 통계(norm stats)를 직접 점검해 정상임을 확인했으므로 **버그가 아니라 순수한 undertrain**이다. 공식 체크포인트(80% 재현)는 이보다 훨씬 오래 학습된 결과이니, 0.45 epoch로는 애초에 수렴할 수 없는 게 당연하다.

과정에서 얻은 교훈 하나: 처음엔 wandb를 꺼두고 돌리는 바람에 **loss 곡선을 전혀 남기지 못했다.** 이후 `finetune.py`에 `[metrics]` 태그로 step/loss를 직접 print하도록 패치해서, 외부 로깅 없이도 학습 진행 상황을 콘솔에서 확인할 수 있게 고쳤다.

> **정직한 결론:** VLA 평가는 논문값을 재현할 수 있고, 파인튜닝 파이프라인도 정상적으로 구동된다. 하지만 **VLA 파인튜닝은 compute-heavy**해서, 단일 GPU로 돌린 짧은 run은 LIBERO 성능으로 transfer되지 않는다. "체크포인트 실행"과 "VLA를 훈련시켜 쓸 만하게 만드는 것"은 전혀 다른 난이도의 작업이라는 걸 직접 확인한 결과다.

## 관찰 / 한계

- **VLA는 스택 전체를 대체한다**: L1/L2는 "인지 → 우리 IK/제어"였지만 OpenVLA는 인지·IK·grasp 계획 없이 이미지+문장에서 action을 바로 낸다. 컨텍스트 전달 사다리 L1(포인터) → L2(언어 grounding) → **L3(VLA)**가 이걸로 완성된다.
- **캐비엇 (1)**: 위 결과는 OpenVLA의 **자기 벤치(LIBERO) + 공식 파인튜닝 체크포인트** 결과지, 우리 MuJoCo 커스텀 씬이 아니다 — VLA는 학습된 embodiment/action-space 안에서만 유효하다.
- **캐비엇 (2)**: 체크포인트를 실행해 논문값을 재현하는 것과 VLA를 처음부터 훈련시키는 것은 다르다. 후자를 직접 시도해본 결과가 위의 "0.45 epoch → 0%" 정직한 undertrain 사례다.
- **캐비엇 (3, 4)**: 20-ep는 논문 full-eval(500-ep)의 부분집합이고, 어디까지나 시뮬레이션이며 실사 로봇은 아니다.
- **다음 축**: 다른 suite(object/goal/long) 간 실패 양상 비교, Octo 같은 경량 모델과의 직접 대조, [grasp_sota.md](grasp_sota.md)의 grasp SOTA A/B와 묶어 "인지형 grasp vs end-to-end VLA 정책" 관점으로 연결. VLA와 world-model 기반 RL 계열의 패러다임 차이는 [concepts.md](concepts.md) 참고.

## 재현

```bash
# WSL2 ov env (torch2.2 + flash-attn + LIBERO + robosuite)
conda run -n ov bash _wsl/run_ov_eval.sh   # openvla-7b-finetuned-libero-spatial, 첫 실행 시 체크포인트 자동 다운로드(~14GB)
```
