# Grasp Detection

로봇이 물체를 **어디를 어떻게 잡을지** 예측하는 grasp 검출 모델 리뷰. 우리 [bin picking A/B](../grasp_sota.md)에 직접 통합해 top-down heuristic과 비교한 모델들이다.

- [GraspNet-1B (2020)](graspnet.md) — 6-DoF grasp 검출의 기준선(PointNet++ → ApproachNet/OperationNet/ToleranceNet). Windows/WSL에서 CUDA 컴파일·통합.
- [ZeroGrasp (2025)](zerograsp.md) — object-centric reconstruction 기반 SOTA. open-tray A/B에서 학습 grasp 1위(단 GT mask 사용).

## 우리 실측 비교 (6물체 · 5 seed · 평균 clearance)

| 모델 | 연도 | 입력 | walled bin | open tray |
|------|------|------|-----------|-----------|
| top-down heuristic (ours) | — | depth+seg | **63%** | **77%** |
| [ZeroGrasp](zerograsp.md) | 2025 | RGB-D + GT mask | 20% | **50%** (학습 모델 1위) |
| graspness / GSNet | 2021 | depth (ME) | 27% | 40% |
| EconomicGrasp | 2024 | depth (ME) | 27% | 33% |
| Contact-GraspNet | 2021 | depth | **33%** | 20% |
| [GraspNet-1B](graspnet.md) | 2020 | depth | 11% | 27% |
| HGGD | 2023 | RGB-D heatmap | ~0% | ~0% |

두 줄 요약: **(1) 2020→2025 5년치 SOTA 6종 전부, 우리 씬에 co-tune된 단순 휴리스틱을 못 넘었다** — "논문 grasp score"와 "이 그리퍼·이 제어로 실제 성공" 사이엔 gripper co-tuning·실행이라는 별도의 층이 있다. **(2) 벽 하나가 순위를 뒤집는다** — walled→open에서 ZeroGrasp 20→50%: 6-DoF tilt 강점은 환경 제약(벽이 강제하는 수직 접근) 앞에서 쉽게 죽는다. 전체 6모델 A/B와 영상은 [Grasp SOTA](../grasp_sota.md).
