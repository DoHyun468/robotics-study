# Grasp Detection

로봇이 물체를 **어디를 어떻게 잡을지** 예측하는 grasp 검출 모델 리뷰. 우리 [bin picking A/B](../grasp_sota.md)에 직접 통합해 top-down heuristic과 비교한 모델들이다.

- [GraspNet-1B (2020)](graspnet.md) — 6-DoF grasp 검출의 기준선(PointNet++ → ApproachNet/OperationNet/ToleranceNet). Windows/WSL에서 CUDA 컴파일·통합.
- [ZeroGrasp (2025)](zerograsp.md) — object-centric reconstruction 기반 SOTA. open-tray A/B에서 학습 grasp 1위(단 GT mask 사용).
