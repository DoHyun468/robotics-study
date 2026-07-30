# GraspNet-1B

## 한 줄 요약
> point cloud에서 곧장 6-DoF grasp(접근방향·회전·깊이·폭·점수)를 예측하는 PointNet++ 기반 grasp 검출기 — ApproachNet/OperationNet/ToleranceNet 세 네트워크로 나눠 대규모(1B 단위 grasp pose) 데이터로 학습한, 학습 기반 grasp 검출의 기준선(CVPR 2020).

## 문제
antipodal·top-down 같은 기하 규칙 기반 grasp 계획은 물체가 기울거나 더미 속에 쐐기처럼 끼면 원리적으로 못 잡는다 — 수직 평행그리퍼는 옆으로 누운 물체의 단면을 애초에 확보할 수 없다. GraspNet-1B는 이 한계를 "안전한 6-DoF grasp pose를 point cloud에서 직접 회귀하는 학습 문제"로 재정의하고, 그걸 학습시킬 만큼 큰 grasp 라벨 데이터셋(1B 단위 grasp pose, 다수 물체, 실제+렌더 장면)을 함께 낸 논문이다.

## 방법
- **backbone**: PointNet++로 장면 point cloud를 인코딩.
- **ApproachNet**: 각 점에서 approach 방향(접근 축) 후보를 제안.
- **OperationNet**: 그 approach 축을 기준으로 in-plane 회전·grasp depth·폭·grasp score를 회귀.
- **ToleranceNet**: 예측한 grasp이 pose perturbation에 얼마나 강건한지(허용오차)를 함께 추정.
- 세 네트워크가 이어져 point cloud 하나에서 곧장 순위 매겨진 6-DoF grasp pose 후보 집합을 낸다 — antipodal 조건 같은 기하 규칙을 사람이 짜 넣지 않는다는 게 핵심 차이.

## 결과
논문이 보고하는 GraspNet-1B 자체 벤치마크 수치는 우리가 재현하지 않았으므로 다루지 않는다. 대신 우리가 직접 측정한 건 **이 모델을 우리 MuJoCo bin-picking 씬에 그대로 꽂았을 때의 clearance rate**다. 6물체·5 seed 기준:

| 조건 | GraspNet-1B (learned) | top-down heuristic (ours) |
|---|---|---|
| walled bin | **~11%** | 63% |
| open tray(벽 제거) | **27%** | 77% |

두 조건 모두 학습 기반 6-DoF grasp이 우리 단순 휴리스틱에 졌다. 이후 같은 씬에서 2020→2025 오픈 후속 모델 5개(Contact-GraspNet, graspness, EconomicGrasp, ZeroGrasp, HGGD)를 추가로 실측했을 때도 전부 휴리스틱을 못 넘었지만, GraspNet-1B(11/27%)보다는 대체로 높은 clearance를 냈다 — GraspNet-1B가 이 계보의 출발점(baseline)이라는 위치가 우리 실측에서도 그대로 드러난다. 자세한 6모델 비교는 [Grasp SOTA A/B](../grasp_sota.md)에서 다룬다.

## 내 실습 연결
GraspNet-1B는 우리가 **가중치까지 받아 실제로 돌린** 유일한 학습 기반 grasp 모델이다. Windows에서 pointnet2 CUDA 연산을 직접 컴파일했고(VS2022 + nvcc 12.4, sm_89, RTX 4090), graspnetAPI의 무거운 의존성은 우회(구버전 numpy 소스빌드를 피하고 `pred_decode`만 사용)해서 공개 체크포인트로 실제 추론이 되는 걸 확인했다(씬 point cloud → 약 758개 grasp 후보, top score 0.9). 같은 모델을 WSL2(Ubuntu, 동일 4090)에서도 별도 conda 환경으로 재현해, Windows에서 못 도는 다른 모델들(MinkowskiEngine·octree 계열)과 나란히 비교할 수 있는 하네스를 만들었다.

통합은 [Manipulation](../manipulation.md)의 bin-picking 파이프라인을 그대로 재사용했다 — 카메라 point cloud → GraspNet 추론 → world 좌표 변환 → feasibility 필터(폭·downward·bin 내부·IK reachable) → 동일한 IK/실행 루프로 6-DoF grasp을 실행. **learned grasp이 실제로 물체를 집어 place까지 성공**하는 건 확인했지만(최상단 물체 기준), 6물체 clutter 전체를 비우는 clearance rate는 walled 11%/open 27%로 휴리스틱(63%/77%)에 못 미쳤다. 원인은 두 갈래로 분해했다: (1) bin 벽이 near-vertical approach를 강제해 GraspNet의 tilt(6-DoF) 강점이 무력화됐고(급경사 approach는 벽 충돌로 실행 불가), (2) 예측된 grasp이 우리 그리퍼의 engagement 깊이·폭이나 position 제어에 co-tune돼 있지 않아, 무너진 더미의 낮은 물체에서 얕게 잡혀 미끄러지거나 헛집었다. 벽을 없앤 open tray A/B에서 GraspNet 쪽만 11→27%(2.4배)로 크게 오른 게 (1)번 가설을 뒷받침했다.

<img src="../_static/bin_pick_montage.png" style="width:100%;max-width:820px;border-radius:8px">

*시도별 lift 순간 — heuristic부터 GraspNet 통합까지.*

## 한 줄 평 / 한계
PointNet++ → ApproachNet/OperationNet/ToleranceNet 구조와 1B 단위 grasp 라벨 데이터셋은 이후 오픈 후속 모델(Contact-GraspNet, EconomicGrasp, ZeroGrasp)이 갈아엎고 넘어선 출발점이라는 역사적 위치가 뚜렷하다 — 우리가 실측한 계보 표에서도 GraspNet-1B가 학습 모델 중 가장 낮은 clearance(11/27%)를 기록해 그 위치를 확인했다. 다만 우리 벤치에서 GraspNet-1B가 휴리스틱에 진 걸 "모델이 나쁘다"로 읽으면 안 된다 — 우리 씬은 합성 clean box·단일 뷰·무텍스처라는 도메인갭이 있고, 무엇보다 grasp이 우리 그리퍼·position 제어에 co-tune되지 않았다는 실행(제어) 쪽 제약이 컸다. Windows에서 CUDA 확장을 직접 컴파일해 실제 추론을 돌리고 MuJoCo 파이프라인에 end-to-end로 꽂아본 경험 자체가, "논문 grasp score가 높다"와 "우리 로봇이 실제로 성공한다" 사이에 gripper co-tuning·제어라는 별도의 층이 있다는 걸 직접 확인시켜준 지점이었다.
