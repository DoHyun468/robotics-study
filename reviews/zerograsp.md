# ZeroGrasp (2025)

*보이지 않는 부분을 재구성하고 잡는다 — reconstruction 기반 grasp SOTA*

Iwase 외, *ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping*, CVPR 2025 (arXiv 2504.10857). 카테고리 개요와 6모델 실측 비교는 [Grasp 리뷰 허브](grasp.md)·[Grasp SOTA A/B](../grasp_sota.md).

## 한 줄 요약

> RGB-D + per-object 인스턴스 마스크를 받아 **물체별 3D 재구성(octree)** 을 먼저 하고 그 표면 위에서 grasp을 예측하는 CVPR'25 모델 — "가려서 안 보이는 뒷면까지 복원한 뒤 잡는다"는 reconstruction-first 접근. 우리 WSL2 6모델 A/B에서 open tray 학습 모델 중 **1위(50%)**를 실측했다.

## 문제 — 부분 관측에서 잡기, 그리고 "2020 이후 SOTA가 있나"

**논문의 문제의식**: 단일 뷰 RGB-D는 물체의 앞면만 보여준다. 기존 grasp 검출기([GraspNet-1B](graspnet.md) 계열)는 이 **부분(partial) 관측** 위에서 곧장 grasp을 뽑는데, 그러면 보이지 않는 뒷면·가림(occlusion) 영역의 grasp 기회를 놓치거나 위험을 과소평가한다. ZeroGrasp의 답: **잡기 전에 형상을 완성하라** — 가려진 부분을 재구성하고, 완성된 표면 위에서 grasp을 예측하면 두 문제가 함께 좋아진다는 것.

**우리의 문제의식**: "2020 GraspNet-1B 이후 오픈 SOTA가 실재하나"를 확인하려고 오픈 후속 모델 6개를 전부 직접 돌린 실측의 마지막 축이 ZeroGrasp다. 다른 모델들과 달리 object-centric 구조라 입력에 **per-object GT 인스턴스 마스크가 필수**고(실전이라면 SAM류 프론트엔드가 필요하다는 뜻), octree/ocnn CUDA 확장이 Windows에서 빌드가 안 돼 별도 환경부터 세워야 했다.

## 방법

### 논문 쪽 — 재구성과 grasp을 한 모델에서

- **구조**: RGB + depth + per-object 인스턴스 마스크 → **octree 기반 조건부 VAE(CVAE)** 가 물체별 3D 형상을 재구성 → 재구성된 표면 위에서 grasp 예측. octree 표현이라 고해상도 형상을 메모리 효율적으로 다룬다(near real-time 주장).
- **다중 물체 추론**: **multi-object encoder**로 물체들을 함께 인코딩하고, **3D occlusion field**(레이캐스팅으로 계산한 상호·자기 가림 정보)를 넣어 "어떤 영역이 왜 안 보이는지"를 명시적으로 모델링 — clutter에서 물체 간 공간 관계를 추론하는 게 재구성·grasp 모두에 이득이라는 게 핵심 주장.
- **데이터**: 자체 합성 데이터셋 — Objaverse-LVIS **12K 물체**에 대해 **photo-realistic 이미지 100만 장 + 물리 검증된 grasp 주석 113억 개**. [GraspNet-1B](graspnet.md)의 "라벨은 계산으로 만든다" 노선을 렌더링 품질·물체 다양성 쪽으로 확장한 셈이다.
- **평가(원문)**: GraspNet-1B 벤치마크에서 SOTA, 실로봇 실험으로 novel object 일반화 시연.

### 우리 쪽 — 동일 하네스 통합

- WSL2(Ubuntu, RTX 4090)에 전용 conda env `zg`: torch 2.2.0+cu121 + ocnn 2.2.5 / dwconv / octree_feature_extractor. pybind11-nvcc 충돌 회피를 위해 gcc11 고정 + `--no-build-isolation`, ocnn·setuptools(<81) 버전 핀 고정으로 빌드 통과.
- 함정 하나: EGL 렌더러는 torch import **전에** 만들어야 WSL에서 프로세스가 안 멈춘다 — 드라이버가 torch를 `load_net()` 안에서 지연 import하도록 순서를 맞췄다.
- 다른 5개 모델(GraspNet-1B · Contact-GraspNet · graspness · EconomicGrasp · HGGD)과 **완전히 같은 하네스**: 같은 MuJoCo bin 씬, 같은 IK, 같은 실행 루프, 같은 feasibility 필터(width·downward·interior·reachable). 바뀌는 건 grasp 검출 알고리즘뿐.

## 결과

6물체 · 5 seed, walled(벽 bin) / open tray(벽 제거) 두 조건, 평균 clearance rate:

| method | 발표 | modality | walled | open |
|---|---|---|---|---|
| top-down heuristic (ours) | — | depth+seg | 63% | 77% |
| **ZeroGrasp** | CVPR'25 | RGB-D + GT mask, recon | 20% | **50%** |
| Contact-GraspNet | RSS'21 | depth | 33% | 20% |
| graspness / GSNet | ICCV'21 | depth (ME) | 27% | 40% |
| EconomicGrasp | ECCV'24 | depth (ME) | 27% | 33% |
| [GraspNet-1B](graspnet.md) baseline | CVPR'20 | depth | 11% | 27% |
| HGGD | RA-L'23 | RGB-D heatmap | ~0% | ~0% |

open tray에서 ZeroGrasp(50%)는 6개 학습 모델 중 1위 — 2위 graspness(40%), 이어서 EconomicGrasp(33%) · Contact-GraspNet(20%) · GraspNet-1B(27%) 순. 다만 walled에서는 20%로 뚝 떨어져 하위권이다: 벽이 근접-수직 approach를 강제해 6-DoF tilt 강점을 무력화한 결과다. open이든 walled든, top-down heuristic(63/77%)은 여전히 이기지 못한다.

<video src="../_static/zg_open_s.mp4" controls loop muted playsinline style="width:100%;max-width:720px;border-radius:8px"></video>

*open tray 4/6*

## 내 실습 연결

전체 6모델 apples-to-apples 비교, 진단, 재현 커맨드는 [Grasp SOTA A/B](../grasp_sota.md)에 정리했다. ZeroGrasp는 그 표의 마지막 줄이자 "2020 GraspNet 이후 오픈 SOTA가 있나"라는 질문에 대한 가장 직접적인 답이다 — **있다**, 공정한(벽 없는) 과제에서는 최신 모델이 학습 계열 중 최고다. 또 "재구성 먼저, grasp은 그 위에서"라는 구조는 이 사이트의 [perception 파이프라인](../perception.md)(depth → point cloud → 형상 정합 → 타깃)과 정확히 같은 설계 사상이다 — 우리가 마커리스 ICP에서 손으로 만든 "형상을 알고 잡기"를, ZeroGrasp는 학습된 재구성으로 일반 물체까지 확장한 셈이다.

## 한 줄 평 / 한계

**한 줄 평.** "보이지 않는 부분을 복원한 뒤 잡는다"는 reconstruction-first 설계로 GraspNet 계보의 5년을 갱신한 SOTA — 그리고 우리 실측에서 그 갱신이 실재함을(open tray 1위) 확인한 모델. 다만 아래 두 조건을 같이 봐야 공정하다.

**한계.**
- **GT 마스크 의존**: 다른 5개 모델과 달리 per-object GT 마스크를 받는다 — 실전이라면 SAM류 세그멘테이션이 앞단에 필요하다. (비교에 쓴 depth도 전부 렌더 GT라 관대하지만, 모든 모델에 동일하게 관대하므로 모델 간 상대 비교는 유효하다.)
- **여전히 휴리스틱을 못 넘음**: open tray 1위(50%)라도 우리 씬에 co-tune된 top-down heuristic(63/77%)엔 진다 — "학습 SOTA면 자동으로 이긴다"는 낙관은 이 벤치에서 기각된다.
- **환경 제약 취약성**: walled 20% → open 50%의 낙폭 자체가, 6-DoF tilt가 진짜 강점이되 환경 제약(벽) 앞에서 쉽게 죽는다는 걸 가장 선명하게 보여준 사례다.
- **합성→실물 격차**: 학습이 합성 데이터(100만 장) 기반이라, 논문의 실로봇 시연 밖 조건(우리 씬 포함)에서의 성능은 도메인갭·실행 co-tuning에 크게 좌우된다.
