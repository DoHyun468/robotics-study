# Gaussian Splatting

*3DGS를 "예쁘게"가 아니라 "정확하게" — 리콘랩스에서 재현·검증·프로덕션 채택을 직접 결정했던 논문들. 전부 원문 PDF를 직접 읽고 정리했다.*

실무 맥락: 리콘랩스 3D 캡처 서비스에서 **mip-splatting 재현을 Baseline**으로 두고 품질 비교 끝에 **RaDe-GS를 채택·통합**했다(Dockerfile·kubeflow·CUDA 재빌드). 2DGS는 기하 정확 GS의 기준선으로 재현·검증했다. 각 리뷰의 "내 실습 연결"에 채택/비채택의 이유를 적었다.

| 리뷰 | 축 | 한 줄 |
|---|---|---|
| [Mip-Splatting](mip-splatting.md) (CVPR 2024) | 앨리어싱 | 3D smoothing + 2D Mip 필터 — 스케일이 바뀌어도 무너지지 않는 splatting |
| [2DGS](2dgs.md) (SIGGRAPH 2024) | 표면 기하 | 가우시안을 2D 디스크로 눕혀 multi-view 일관 기하를 얻는다 |
| [RaDe-GS](rade-gs.md) (2024) | depth 정밀 | 3D 가우시안을 유지한 채 ray별 depth를 닫힌형으로 — 표현력과 기하의 양립 |
| [3DGS-MCMC](gs-mcmc.md) (NeurIPS 2024) | 최적화 관점 | clone/split 휴리스틱을 MCMC 샘플링으로 — 초기화 의존을 지운다 |

**선택의 축** — 렌더링 품질만 보면 3DGS 계열은 이미 충분하다. 문제는 ① 스케일 변화(mip), ② 기하 정확도(2DGS/RaDe-GS), ③ 최적화 안정성(MCMC)이고, 계측·로보틱스로 갈수록 ②가 지배한다 — 복원 기반 치수측정(GT 대비 −1.0/+0.3 mm)이 가능했던 것도 depth가 정확한 표현을 골랐기 때문이다.
