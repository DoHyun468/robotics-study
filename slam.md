# SLAM — Localization & Mapping

이 사이트의 [perception](perception.md) 파이프라인은 **고정 카메라의 단일 시점 RGB-D 스냅샷**을 매번 새로 찍어 물체를 인지한다. 카메라가 움직이지도, 프레임을 시간으로 잇지도 않는다. SLAM은 정확히 그 비어 있는 축 — **움직이는 센서로 "내가 어디 있나(localization)"와 "주변이 어떻게 생겼나(mapping)"를 동시에 푸는 것** — 을 채운다. 이 페이지는 개념 정리(문헌)이고, 우리가 직접 구현·측정한 실측은 별도로 붙인다(아래 §7).

포지셔닝 관점: **3D perception → mapping/localization(SLAM) → spatial AI**. stereo matching·camera calibration·depth 같은 기존 기하 역량과 NeRF/3DGS 재구성 경험이 **뉴럴 SLAM**에서 한 줄로 이어진다.

## 1. SLAM이 푸는 문제

한 문장: **카메라(또는 LiDAR)가 움직이는 동안, 관측만으로 센서의 궤적 $T_1,\dots,T_n$ 과 장면의 지도 $M$ 을 동시에 추정한다.** 지도가 있으면 위치를 알기 쉽고, 위치를 알면 지도를 쌓기 쉬운 **닭-달걀 문제**라서 "동시에(simultaneous)"가 핵심이다.

우리 현재 인지와의 대비:

| | 현재 (single-shot 인지) | SLAM |
|---|---|---|
| 카메라 | 고정, GT extrinsic | **움직임**, extrinsic을 스스로 추정 |
| 시간축 | 없음(매 프레임 독립) | **프레임 간 트래킹·누적** |
| 산출물 | 물체 pose 1장 | **궤적 + 지속 지도** |
| 오차 | 캘리브·depth 오차(정적) | +**drift**(오차가 시간에 누적) |

→ 왜 우리에게 필요한가: eye-in-hand(손목) 카메라 자기위치추정, 멀티스텝 조작용 지속 지도, 마커·GT 없는 markerless extrinsic. 단일 샷 grasp 인지에서 **공간 인지형 조작**으로 넘어가는 다리.

## 2. 고전 SLAM 해부 — 프론트엔드 · 백엔드 · 루프클로저

- **프론트엔드 (tracking / odometry)**: 연속 프레임 사이의 상대 모션을 추정. 두 갈래 —
  - **특징 기반(feature-based)**: ORB 같은 특징점 추출·매칭 → 에피폴라/PnP로 모션. 대표 **ORB-SLAM** 계열. 조명·텍스처에 견고, 재관측·재localization 쉬움.
  - **직접법(direct)**: 특징 없이 **픽셀 밝기(photometric error)** 를 직접 최소화. 대표 **DSO·LSD-SLAM**. 텍스처 적은 곳에 강점, 미세 모션에 정밀.
- **백엔드 (optimization)**: 프론트엔드 추정을 전역적으로 다듬어 **drift**를 줄인다.
  - **필터 기반**(EKF/MSCKF) vs **그래프 기반**(pose-graph, **bundle adjustment**). 현대 시각 SLAM은 대부분 그래프+BA(Ceres/GTSAM/g2o로 비선형 최소제곱).
- **루프 클로저 (loop closure)**: 예전에 본 곳을 다시 알아보고(장소 인식, e.g. DBoW) 그 제약을 그래프에 넣어 **누적 drift를 한 번에 접어** 전역 일관성 회복.
- **지도 표현 (map representation)**: sparse 포인트 → dense 포인트/**surfel** → 볼류메트릭 **TSDF/occupancy**(KinectFusion) → 최신 **NeRF/3D Gaussian**.

## 3. 센서·방식 분류

| 축 | 옵션 | 특징 |
|---|---|---|
| 입력 | mono / **stereo** / **RGB-D** / VIO(+IMU) / LiDAR | mono는 **스케일 모호**(절대 크기 모름), stereo·RGB-D·LiDAR는 metric scale 확보 |
| 프론트엔드 | 특징 기반 / 직접법 | ORB-SLAM3 vs DSO |
| 밀도 | sparse / semi-dense / **dense** | 조작·재구성엔 dense/RGB-D가 유리 |
| 백엔드 | 필터 / 그래프+BA | 현대 표준은 그래프+BA |

- **RGB-D SLAM**(우리 셋업과 직결): depth로 스케일이 바로 나오고, **frame-to-model** 트래킹 + TSDF 융합(KinectFusion·ElasticFusion)이 정석. MuJoCo가 RGB-D를 그대로 주므로 우리 실측의 출발점.
- **VIO/VINS**(VINS-Mono, OKVIS): IMU를 붙여 저텍스처·빠른 모션에 견고. 로봇·드론·AR의 실전 주류.

## 4. 현대 · 학습 기반 · 뉴럴 SLAM (차별화 축)

고전(hand-crafted) → 딥러닝(학습된 특징·깊이·flow) → **뉴럴 표현(NeRF/3DGS)** 으로 흐름이 이동 중.

- **학습형 VO/SLAM**: **DROID-SLAM**(dense optical flow + 미분가능 BA)처럼 end-to-end로 견고성↑. 학습된 특징/깊이/flow가 고전 모듈을 대체.
- **NeRF SLAM**(iMAP·NICE-SLAM): 장면을 신경장으로 표현, 렌더링 오차로 트래킹+맵핑. 사실적이지만 느림.
- **3D Gaussian Splatting SLAM**(SplaTAM·GS-SLAM·MonoGS): 장면을 3D 가우시안으로 표현 → **미분가능 rasterization으로 실시간** 트래킹+photorealistic 맵. NeRF의 속도 한계를 넘어 현재 최전선.
- **파운데이션/기하 사전학습**: DUSt3R·VGGT 등 "이미지 쌍→3D"를 바로 뱉는 모델이 SfM/SLAM 프론트엔드를 재편 중.

→ 이 갈래가 우리 NeRF/3DGS 재구성 배경과 정확히 겹친다. 표현(representation) 관점의 연결은 [Concepts](concepts.md)·[World Models](world-models/predictive.md)(공간/예측 표현)와도 이어진다.

## 5. 평가 — 무엇을 숫자로 보나

- **ATE (Absolute Trajectory Error)**: 추정 궤적을 GT에 정렬(Sim3/SE3) 후 위치 RMSE — 전역 일관성.
- **RPE (Relative Pose Error)**: 국소 상대 모션 오차 — drift 율(odometry 품질).
- **맵 품질**: 재구성 표면의 정확도/완성도, 렌더링 PSNR(뉴럴 SLAM).
- **표준 데이터셋**: **TUM RGB-D**, **EuRoC**(VIO), **KITTI**(야외/LiDAR).

우리 강점: **MuJoCo가 GT 카메라 궤적·기하를 공짜로** 준다 → 추정 궤적을 GT와 직접 대조해 ATE/RPE를 정량화. [error budget](perception.md) 페이지에서 캘리브 오차를 base 좌표까지 전파했던 것과 같은 **정직 정량** 방식을 SLAM에 그대로 적용한다.

## 6. 학습 로드맵 & 자료

가장 체계적인 기준: **[changh95/visual-slam-roadmap](https://github.com/changh95/visual-slam-roadmap)** (11레벨, 수학→고전→RGB-D→딥러닝→VIO→…→월드모델, 2024–2026 논문·DUSt3R/VGGT/3DGS-SLAM 포함).

- **Tier 0 · 수학/다중뷰 기하**: [TUM Multiple View Geometry (Cremers)](https://cvg.cit.tum.de/teaching/online/mvg) · Cyrill Stachniss Photogrammetry
- **Tier 1 · 고전 SLAM**: [Stachniss — Intro to SLAM](https://www.youtube.com/watch?v=NW8hUcbjuIQ) · [SLAM-Resources-for-Beginner](https://github.com/Taeyoung96/SLAM-Resources-for-Beginner)(slambook 포함) · [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- **Tier 2 · 현대/뉴럴**: [NeRF·3DGS SLAM 서베이 (Tosi)](https://fabiotosi92.github.io/files/survey-slam.pdf) · [awesome-NeRF-and-3DGS-SLAM](https://github.com/3D-Vision-World/awesome-NeRF-and-3DGS-SLAM)

공부 순서: Tier 0(필요분) → Tier 1 이론 + ORB-SLAM3 돌려보기 → Tier 2 서베이 + 우리 셋업에서 RGB-D/3DGS SLAM 실측.

## 7. 우리 실측 계획 (별도 · 진행 예정)

문헌과 분리해, MuJoCo 위에서 단계별로 직접 구현하고 **GT로 벤치마크**한다:

- **S0** 움직이는 카메라 RGB-D+GT 시퀀스 생성 → **S1** RGB-D 오도메트리(ICP scan-to-scan + ORB/PnP, ATE/RPE) → **S2** TSDF 맵핑 → **S3** ORB-SLAM3 벤치마크 + 미니 pose-graph(루프클로저) → **S4** 뉴럴 3DGS SLAM.

각 단계 결과(영상·궤적·오차 그래프)는 완성되는 대로 실측 섹션에 추가한다. 기존 [perception](perception.md)의 캘리브·ICP·RGB-D 프리미티브를 그대로 재사용한다.

---

*이 페이지는 SLAM 개념·논문의 정성적 정리(문헌)이며, 이 프로젝트에서 직접 측정한 수치가 아니다. 실측 결과는 §7의 실측 트랙에서 별도로 다룬다.*
