# Calibration 리뷰 허브

*석사 계측 과제의 두 기둥 — 내부는 ray로, 외부는 1D로 — 그 이론적 근거가 된 논문들.*

[Camera Geometry & Calibration 시리즈](../vision/index.md)의 8·9·10장이 표준 해법(Brown 왜곡 + Zhang + PnP)을 다룬다면, 여기서는 표준이 모자랄 때 꺼내는 두 도구의 원전을 읽는다.

| 리뷰 | 질문 | 원전 |
|---|---|---|
| [Ray (Generic) Calibration](ray-calibration.md) | 다항식 왜곡 모델이 렌즈를 못 따라가면? → **모델을 버리고 픽셀별 광선을 잰다** | Grossberg & Nayar (ICCV 2001) · Sturm & Ramalingam (ECCV 2004) · Schöps et al. (CVPR 2020) |
| [1D Target Calibration](1d-calibration.md) | 현장에서 대형 2D 보드를 못 쓰면? → **막대 하나로 언제 캘리브레이션이 가능한가** | Zhang (TPAMI 2004) |

**과제와의 연결** — 선체블록 계측 시스템에서 내부 파라미터는 모니터 패턴 기반 ray calibration으로(실험실 24장), 외부 파라미터는 화이트보드·1D 타깃으로(현장 50장) 추정했다. 3차 현장 계측: W.D 3.7~17.2 m, 광파기 대비 평균 2.46 mm(화이트보드) / 2.68 mm(1D), 5 mm 이내 100% / 95.65%.
