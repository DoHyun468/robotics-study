# Digital Twin · Scan-to-Sim

실제 공간·물체를 스캔해서 **물리 시뮬레이션이 돌아가는 가상 자산(sim-ready asset)**으로 만드는 파이프라인을 다룬다. 3D 복원(NeRF/GS/photogrammetry)까지는 "보이는 것"의 문제지만, 시뮬레이터에 들어가는 순간 전혀 다른 요구가 붙는다 — 충돌 계산이 가능한 메시, 질량·마찰 같은 물리 속성, 그리고 움직이는 부품의 관절 구조.

복원 → 시뮬 사이의 간극을 세 조각으로 나눠 정리한다.

| 페이지 | 질문 | 핵심 |
|---|---|---|
| [OpenUSD 기초](openusd.md) | 디지털 트윈 씬은 어떤 포맷 위에 서나? | Stage/Prim/Layer/Composition, UsdPhysics 스키마, sim-ready 체크리스트, MJCF 대응표 |
| [관절 구조 추정](articulation.md) | 정지 스캔에 없는 "움직임" 정보를 어떻게 얻나? | screw axis 역산 수식, 다중 상태 관측의 필연성, Shape2Motion→ANCSH→Ditto→PARIS 계보 |
| [GS → sim-ready 메시](simready-mesh.md) | 가우시안 복원을 물리 엔진에 넣으려면? | 메시 추출(TSDF/Poisson/SuGaR), convex decomposition(V-HACD/CoACD), 물리 속성 부여 |

**이 파트의 한 줄 요약**: 스캔은 기하(geometry)를 주지만, 시뮬레이션은 **기하 + 물리 + 구조(관절)**를 요구한다. 그 차액을 채우는 기술들이 Scan-to-Sim이다.

**기존 파트와의 연결**: 복원 품질 자체는 [GS 리뷰](../reviews/gs.md)(mip-splatting·2DGS·RaDe-GS)와 [feed-forward 복원](../reviews/feedforward.md)(DUSt3R·VGGT)이 담당한다. 여기서 만든 가상 환경의 소비자는 [World Models](../world-models/four-families.md)와 [VLA](../vla.md) — 시뮬에서 배우는 지능이다.
