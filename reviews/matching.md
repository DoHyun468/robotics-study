# Feature Matching & SfM

*대응점은 3D 비전의 화폐다 — 석사 주제(wide-baseline 선체블록 대응)가 서 있던 지반의 원전들, 그리고 그 대응점을 소비하는 SfM의 표준(COLMAP)까지. 전부 원문 PDF를 직접 읽고 정리했다.*

내 석사 파이프라인은 정확히 이 논문들의 조합이었다 — **SuperGlue·DKM·LoFTR 앙상블**로 keypoint 매칭 → 클러스터링 keypoint enhancement로 ROI → U-Net guided mask → **COTR** 정밀 매칭(평균 오차 29.1→2.23 px, PCK-15px 34.2→84.6%). 각 리뷰의 "내 실습 연결"에서 무엇이 통했고 무엇이 무너졌는지를 적었다.

먼저 [**SuperGlue vs LoFTR vs DKM 비교**](matching-compare.md)에서 세 계열의 장단·신뢰도 신호·"키포인트"의 정체 차이를 한 페이지로 정리했다.

| 리뷰 | 계열 | 한 줄 |
|---|---|---|
| [SuperGlue](superglue.md) (CVPR 2020) | sparse + 학습 매처 | 검출된 keypoint 위에서 attentional GNN + optimal transport로 "함께 맞는" 매칭 |
| [LightGlue](lightglue.md) (ICCV 2023) | sparse, SG 후속 | 쉬운 쌍은 일찍 멈춘다 — 적응형 깊이·pruning으로 SG 대비 2배+, 정확도는 더 위 |
| [LoFTR](loftr.md) (CVPR 2021) | semi-dense, detector-free | 검출기를 버리고 transformer로 coarse→fine — 저텍스처에서 검출 실패 자체를 우회 |
| [Efficient LoFTR](eloftr.md) (CVPR 2024) | semi-dense, LoFTR 후속 | aggregated attention + 2단 정제 — 정확도를 올리며 sparse급 속도(27ms) |
| [DKM](dkm.md) (CVPR 2023) | dense | Gaussian Process 전역 매처 + certainty — 픽셀 전부를 맞추고 신뢰도로 거른다 |
| [RoMa](roma.md) (CVPR 2024) | dense, DKM 후속 | frozen DINOv2 coarse + 전용 fine — 극단 조건(WxBS +36%)을 뚫는 현 dense SOTA |
| [COTR](cotr.md) (ICCV 2021) | query 기반 | "이 좌표의 대응은 어디인가"를 함수로 — 계측점처럼 임의 지정점에 직접 답한다 |
| [CAPS](caps.md) (ECCV 2020) | query·약지도 | 카메라 포즈만으로 descriptor 학습 — epipolar 손실의 원형, 내 석사 베이스라인 |
| [COLMAP](colmap.md) (CVPR 2016) | incremental SfM | 그 대응점들을 소비해 포즈·구조를 세우는 지난 10년의 표준 |
| [GLOMAP](glomap.md) (ECCV 2024) | global SfM | translation averaging을 버리고 위치+포인트 동시 최적화 — COLMAP 정확도에 수십 배 속도 |

**계보 감각** — sparse(SuperGlue)에서 semi-dense(LoFTR), dense(DKM)로 갈수록 "어디를 매칭할지"의 결정이 검출기에서 모델로 넘어온다. COTR은 축이 다르다(질의 기반). 그리고 이 파이프라인 전체를 feed-forward 회귀로 흡수하려는 흐름이 [DUSt3R 계열](feedforward.md)이다.

*(참고: Agisoft Metashape는 산업에서 널리 쓰이지만 알고리즘을 논문으로 공개한 적이 없는 상용 비공개 소프트웨어라 리뷰 대상이 없다.)*
