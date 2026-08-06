# Camera Geometry & Calibration

*카메라 기하 기초부터 캘리브레이션, 그리고 ray·1D 타깃 캘리브레이션 논문까지 — 석사 계측 과제와 robotics-lab 실측의 이론 백본을 한 줄기로 정리한다.*

이 시리즈는 다크 프로그래머 블로그의 고전 커리큘럼([카메라 캘리브레이션](https://darkpgmr.tistory.com/32) 및 컴퓨터 비전 Geometry 연재)과 같은 순서를 따르되, 설명과 그림은 전부 새로 썼다. 각 장의 수식은 내가 실제로 코드로 구현하며 확인한 것들이고, 장마다 "내 실측과의 연결"을 붙였다.

## 목차

| # | 주제 | 한 줄 |
|---|---|---|
| 1 | [좌표계](coordinates.md) | world → camera → normalized → pixel, 변환 체인 |
| 2 | [Homogeneous Coordinates](homogeneous.md) | 평행이동·투영을 행렬 하나로 만드는 언어 |
| 3 | [2D 변환](transforms-2d.md) | rigid → similarity → affine → projective 위계 |
| 4 | [Homography](homography.md) | 평면이 만드는 8-DoF 사영 변환 — 캘리브레이션의 재료 |
| 5 | [3D 변환](transforms-3d.md) | SO(3)·SE(3), 회전 표현들과 실전 함정 |
| 6 | [이미지 투영](imaging-geometry.md) | 핀홀 모델과 K[R\|t] — 모든 것의 중심 식 |
| 7 | [Epipolar Geometry](epipolar.md) | E·F, 대응 탐색을 1D로 줄이는 제약 — 석사 주제의 무대 |
| 8 | [렌즈 왜곡 보정](distortion.md) | Brown 모델, barrel/pincushion, undistort 실무 |
| 9 | [카메라 캘리브레이션 (Zhang)](calibration.md) | 평면 보드 N장으로 K를 푸는 표준 해법 |
| 10 | [Extrinsic Calibration · PnP](extrinsic.md) | 카메라는 어디에 있는가 — 현장 캘리브레이션으로 |

## 논문 리뷰 (Paper Reviews 파트)

- [Vision Ray Calibration](../reviews/ray-calibration.md) — Bartsch et al., Optics Express 2021. LCD 위상 시프트로 픽셀별 광선을 재고 **멀티카메라를 하나의 광선장으로** — 과제의 ray calibration이 딛고 선 논문. (+ CV 쪽 계보: Grossberg & Nayar · Sturm & Ramalingam · Schöps)
- [1D 타깃 캘리브레이션](../reviews/1d-calibration.md) — Zhang TPAMI 2004(존재 정리) + Duan et al., Optics Express 2022(순환 왜곡 보정으로 0.1% 정밀도). 현장에서 막대 하나로 캘리브레이션이 가능한 근거와 실현.

## 이 시리즈가 닿는 곳

- **석사 계측 과제** — ray calibration(내부) + 화이트보드·1D 타깃(외부) → 조선소 W.D 3.7~17.2 m에서 평균 오차 2.46 mm.
- **robotics-lab** — 시뮬 캘리브(RMS 0.37 px) → pose → hand-eye(4.2 mm/0.10°) → IK pick. [Perception](../perception.md) 장 참조.
