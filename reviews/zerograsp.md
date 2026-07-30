# ZeroGrasp

## 한 줄 요약
> RGB-D + per-object 인스턴스 마스크를 입력받아 물체별 octree reconstruction 위에서 grasp을 예측하는 CVPR'25 모델 — 우리 WSL2 6모델 A/B에서 open tray 학습 모델 중 1위(50%)를 실측했다.

## 문제
"2020 GraspNet-1B 이후 오픈 SOTA가 있나"를 확인하려고 오픈 후속 모델 6개를 전부 직접 돌린 실측의 마지막 축이 ZeroGrasp다. 다른 모델들과 달리 ZeroGrasp는 object-centric 구조다 — 씬 전체 point cloud가 아니라 물체 하나하나를 재구성한 뒤 그 표면 위에서 grasp을 뽑는다. 이 구조상 입력에 **per-object GT 인스턴스 마스크가 필수**다(실제 파이프라인이라면 SAM 같은 세그멘테이션 프론트엔드가 앞단에 있어야 한다는 뜻이기도 하다). 게다가 이 모델이 요구하는 octree/ocnn CUDA 확장은 Windows에서 빌드가 안 돼, 별도 환경부터 새로 세워야 했다.

## 방법
- RGB + depth + per-object 인스턴스 마스크 → 물체별 octree reconstruction(ocnn 기반) → 재구성된 표면 위에서 grasp 예측.
- WSL2(Ubuntu, 동일 RTX 4090)에 ZeroGrasp 전용 conda env `zg`를 새로 세웠다: torch 2.2.0+cu121 + ocnn 2.2.5 / dwconv / octree_feature_extractor. pybind11-nvcc 충돌을 피하려 gcc11로 고정하고 `--no-build-isolation`으로 빌드했으며, ocnn과 setuptools(<81) 버전을 핀 고정해야 빌드가 통과했다.
- 공통 함정 하나 더: EGL 렌더러는 torch import **전에** 만들어야 WSL에서 프로세스가 멈추지 않는다 — 드라이버가 torch를 `load_net()` 안에서 지연 import하도록 순서를 맞췄다.
- 다른 5개 모델(GraspNet-1B · Contact-GraspNet · graspness · EconomicGrasp · HGGD)과 완전히 같은 하네스에 꽂았다: 같은 MuJoCo bin 씬, 같은 IK, 같은 실행 루프, 같은 feasibility 필터(width·downward·interior·reachable). 바뀌는 건 grasp 검출 알고리즘뿐이다.

## 결과
6물체 · 5 seed, walled(벽 bin) / open tray(벽 제거) 두 조건, 평균 clearance rate:

| method | 발표 | modality | walled | open |
|---|---|---|---|---|
| top-down heuristic (ours) | — | depth+seg | 63% | 77% |
| **ZeroGrasp** | CVPR'25 | RGB-D + GT mask, recon | 20% | **50%** |
| Contact-GraspNet | RSS'21 | depth | 33% | 20% |
| graspness / GSNet | ICCV'21 | depth (ME) | 27% | 40% |
| EconomicGrasp | ECCV'24 | depth (ME) | 27% | 33% |
| GraspNet-1B baseline | CVPR'20 | depth | 11% | 27% |
| HGGD | RA-L'23 | RGB-D heatmap | ~0% | ~0% |

open tray에서 ZeroGrasp(50%)는 6개 학습 모델 중 1위다 — 2위 graspness(40%), 이어서 EconomicGrasp(33%) · Contact-GraspNet(20%) · GraspNet-1B(27%) 순. 다만 walled에서는 20%로 뚝 떨어져 6개 중 하위권이다: 벽이 근접-수직 approach를 강제해 ZeroGrasp의 6-DoF tilt 강점을 무력화한 결과다. open이든 walled든, top-down heuristic(63/77%)은 여전히 이기지 못한다.

<video src="../_static/zg_open_s.mp4" controls loop muted playsinline style="width:100%;max-width:720px;border-radius:8px"></video>

*open tray 4/6*

## 내 실습 연결
전체 6모델 apples-to-apples 비교, 진단, 재현 커맨드는 [Grasp SOTA A/B](../grasp_sota.md)에 정리했다. ZeroGrasp는 그 표의 마지막 줄이자 "2020 GraspNet 이후 오픈 SOTA가 있나"라는 질문에 대한 가장 직접적인 답이다 — **있다**, 공정한(벽 없는) 과제에서는 최신 모델이 학습 계열 중 최고다.

## 한 줄 평 / 한계
"2020 이후 SOTA는 없다"는 가정을 깨는 실측 증거지만, 두 가지를 같이 봐야 공정하다. (1) 다른 5개 모델과 달리 ZeroGrasp는 per-object GT 마스크를 받는다 — 실제 파이프라인이라면 SAM 같은 세그멘테이션이 앞단에 추가로 필요하다(비교에 쓴 depth도 전부 렌더 GT라 관대하지만, 모든 모델에 동일하게 관대하므로 모델 간 상대 비교는 유효하다). (2) open tray 1위라도 여전히 우리 씬에 co-tune된 단순 top-down heuristic(63/77%)은 못 넘는다 — "학습 SOTA면 자동으로 이긴다"는 낙관은 이 벤치에서 기각된다. walled→open에서 20%→50%로 뛴 낙폭 자체가 "6-DoF tilt는 진짜 강점이지만 환경 제약(벽) 앞에서 쉽게 죽는다"는 걸 가장 선명하게 보여준 사례였다.
