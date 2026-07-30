# GraspNet-1B (2020)

*10억 grasp 라벨 + point cloud에서 곧장 6-DoF grasp — 학습 기반 grasp 검출의 기준선*

Fang, Wang, Gou, Lu, *GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping*, CVPR 2020. 데이터셋·코드 공개(graspnet.net). 카테고리 개요와 6모델 실측 비교는 [Grasp 리뷰 허브](grasp.md)·[Grasp SOTA A/B](../grasp_sota.md).

## 한 줄 요약

> point cloud에서 곧장 6-DoF grasp(접근방향·회전·깊이·폭·점수)를 예측하는 PointNet++ 기반 grasp 검출기 — ApproachNet/OperationNet/ToleranceNet 세 네트워크로 나눠, 함께 공개한 **11억 grasp 라벨** 데이터셋으로 학습한 학습 기반 grasp 검출의 기준선. 우리가 가중치까지 받아 실제로 돌리고 bin-picking 파이프라인에 통합한 모델이다.

## 문제 — 기하 규칙의 천장, 그리고 데이터 부재

antipodal·top-down 같은 기하 규칙 기반 grasp 계획은 물체가 기울거나 더미 속에 쐐기처럼 끼면 원리적으로 못 잡는다 — 수직 평행그리퍼는 옆으로 누운 물체의 단면을 애초에 확보할 수 없다. 그렇다면 "안전한 6-DoF grasp pose"를 데이터에서 직접 배우면 되는데, 2020년 시점의 진짜 병목은 모델이 아니라 **학습시킬 데이터가 없다**는 것이었다 — grasp 라벨은 이미지 분류 라벨처럼 크라우드소싱으로 만들 수 없다(사람이 6-DoF pose를 일일이 그릴 수 없다).

GraspNet-1B의 답은 **2단 기여**다: (1) 라벨을 해석적으로 대량 생산하는 벤치마크, (2) 그 위에서 도는 end-to-end 검출기.

## 방법

### 데이터셋 — 라벨을 "계산"으로 만든다

- **190개 clutter 장면**, 장면당 카메라 2종(RealSense/Kinect)으로 512장씩 → 총 **97,280 RGB-D 이미지**.
- 물체별 정밀 3D 모델과 6-DoF pose 주석이 있으므로, grasp 라벨은 물체 모델 위에서 **force-closure 같은 해석적 판정**으로 자동 생성해 장면에 투영한다 — 장면당 300만~900만 개, 총 **11억 개 이상의 grasp pose 라벨**. 사람이 그린 게 아니라 **기하·역학으로 계산**한 라벨이라 이 규모가 가능했다.
- 평가도 데이터셋에 내장: 예측 grasp을 물체 모델과 대조해 마찰계수별 성공 여부를 판정하는 프로토콜 제공 — 이후 5년간 후속 연구([ZeroGrasp](zerograsp.md) 포함)가 이 벤치 위에서 경쟁하게 된다.

### 검출기 — 세 갈래로 나눈 6-DoF 회귀

grasp pose는 (접근방향, in-plane 회전, 깊이, 폭)으로 자유도가 높아 한 번에 회귀하면 어렵다. GraspNet-1B는 이를 **단계적으로 분해**한다:

- **backbone**: PointNet++로 장면 point cloud를 인코딩.
- **ApproachNet**: 각 점에서 **approach 방향**(접근 축) 후보를 제안 — "어디서 어느 방향으로 들어갈까"를 먼저 정한다.
- **OperationNet**: 정해진 approach 축을 기준으로 **in-plane 회전·grasp 깊이·폭·점수**를 회귀 — "그 방향에서 손목을 어떻게 돌리고 얼마나 깊이 넣을까".
- **ToleranceNet**: 예측 grasp이 pose perturbation에 얼마나 강건한지(**허용오차**)를 추정 — 실행 오차가 있어도 살아남을 grasp을 우대한다.

antipodal 조건 같은 기하 규칙을 사람이 짜 넣지 않고, point cloud 하나에서 순위 매겨진 6-DoF grasp 후보 집합이 곧장 나온다는 게 규칙 기반과의 핵심 차이다. ToleranceNet의 존재는 특히 실용적이다 — 우리 실측에서 확인한 "실행(제어) 오차가 grasp 성패를 가른다"는 문제를 논문도 이미 인지하고 있었다는 뜻.

## 결과 — 우리 실측 중심

논문 자체 벤치마크 수치는 우리가 재현하지 않았으므로 다루지 않는다. 대신 우리가 직접 측정한 건 **이 모델을 우리 MuJoCo bin-picking 씬에 그대로 꽂았을 때의 clearance rate**다. 6물체·5 seed 기준:

| 조건 | GraspNet-1B (learned) | top-down heuristic (ours) |
|---|---|---|
| walled bin | **~11%** | 63% |
| open tray(벽 제거) | **27%** | 77% |

두 조건 모두 학습 기반 6-DoF grasp이 우리 단순 휴리스틱에 졌다. 이후 같은 씬에서 2020→2025 오픈 후속 모델 5개(Contact-GraspNet, graspness, EconomicGrasp, ZeroGrasp, HGGD)를 추가로 실측했을 때도 전부 휴리스틱을 못 넘었지만, GraspNet-1B(11/27%)보다는 대체로 높은 clearance를 냈다 — GraspNet-1B가 이 계보의 출발점(baseline)이라는 위치가 우리 실측에서도 그대로 드러난다. 자세한 6모델 비교는 [Grasp SOTA A/B](../grasp_sota.md).

## 내 실습 연결

GraspNet-1B는 우리가 **가중치까지 받아 실제로 돌린** 유일한 학습 기반 grasp 모델이다. Windows에서 pointnet2 CUDA 연산을 직접 컴파일했고(VS2022 + nvcc 12.4, sm_89, RTX 4090), graspnetAPI의 무거운 의존성은 우회(구버전 numpy 소스빌드를 피하고 `pred_decode`만 사용)해서 공개 체크포인트로 실제 추론이 되는 걸 확인했다(씬 point cloud → 약 758개 grasp 후보, top score 0.9). 같은 모델을 WSL2(Ubuntu, 동일 4090)에서도 별도 conda 환경으로 재현해, Windows에서 못 도는 다른 모델들(MinkowskiEngine·octree 계열)과 나란히 비교할 수 있는 하네스를 만들었다.

통합은 [Manipulation](../manipulation.md)의 bin-picking 파이프라인을 그대로 재사용했다 — 카메라 point cloud → GraspNet 추론 → world 좌표 변환 → feasibility 필터(폭·downward·bin 내부·IK reachable) → 동일한 IK/실행 루프로 6-DoF grasp을 실행. **learned grasp이 실제로 물체를 집어 place까지 성공**하는 건 확인했지만(최상단 물체 기준), 6물체 clutter 전체를 비우는 clearance rate는 walled 11%/open 27%로 휴리스틱(63%/77%)에 못 미쳤다. 원인은 두 갈래로 분해했다: (1) bin 벽이 near-vertical approach를 강제해 GraspNet의 tilt(6-DoF) 강점이 무력화됐고(급경사 approach는 벽 충돌로 실행 불가), (2) 예측된 grasp이 우리 그리퍼의 engagement 깊이·폭이나 position 제어에 co-tune돼 있지 않아, 무너진 더미의 낮은 물체에서 얕게 잡혀 미끄러지거나 헛집었다. 벽을 없앤 open tray A/B에서 GraspNet 쪽만 11→27%(2.4배)로 크게 오른 게 (1)번 가설을 뒷받침했다.

<img src="../_static/bin_pick_montage.png" style="width:100%;max-width:820px;border-radius:8px">

*시도별 lift 순간 — heuristic부터 GraspNet 통합까지.*

## 한 줄 평 / 한계

**한 줄 평.** "grasp 라벨은 사람이 아니라 기하가 만든다"는 데이터셋 설계와, approach→operation→tolerance로 분해한 검출기로 학습 기반 6-DoF grasp의 판을 깐 논문. 이후 오픈 후속 모델들(Contact-GraspNet, EconomicGrasp, [ZeroGrasp](zerograsp.md))이 갈아엎고 넘어선 **출발점**이라는 역사적 위치가 뚜렷하다 — 우리 실측 계보 표에서도 학습 모델 중 가장 낮은 clearance(11/27%)로 그 위치가 그대로 드러났다.

**한계 (우리 실측이 말해주는 것 포함).**
- **벤치 점수 ≠ 로봇 성공**: 우리 벤치에서 휴리스틱에 진 걸 "모델이 나쁘다"로 읽으면 안 된다 — 합성 clean box·단일 뷰·무텍스처라는 도메인갭에, **grasp이 우리 그리퍼·position 제어에 co-tune되지 않았다**는 실행 쪽 제약이 겹친 결과다. "논문 grasp score가 높다"와 "우리 로봇이 실제로 성공한다" 사이엔 gripper co-tuning·제어라는 별도의 층이 있다.
- **환경 제약에 취약**: 6-DoF tilt 강점은 bin 벽 같은 환경 제약 하나로 쉽게 무력화된다(11%→27% A/B가 직접 증거).
- **grasp 검출까지만**: 검출 이후의 실행(모션·힘 제어·재시도)은 범위 밖 — clearance를 결정하는 나머지 절반이 여기 있다는 걸 우리 실측이 보여줬다.
