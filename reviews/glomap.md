# GLOMAP — Global Structure-from-Motion Revisited (ECCV 2024)

*"translation averaging을 버려라" — global SfM이 20년간 정확도에서 밀린 원인을 translation averaging 한 단계로 지목하고, 그 자리를 카메라 위치+3D 포인트 동시 추정(global positioning)으로 갈아끼워 COLMAP급 정확도를 수십 배 빠르게 달성한 시스템 논문. 원문 PDF 전 페이지(보충자료 포함)를 직접 읽고 정리했다.*

Linfei Pan, Dániel Baráth, Marc Pollefeys, Johannes L. Schönberger, *Global Structure-from-Motion Revisited*, ECCV 2024. ([arXiv:2407.20219](https://arxiv.org/abs/2407.20219) · [GitHub: colmap/glomap](https://github.com/colmap/glomap)) — 제목부터 COLMAP 논문("Structure-from-Motion Revisited", CVPR 2016)의 global 버전 선언이며, 실제로 COLMAP 생태계 안에서 배포된다(대응 검색·DB 포맷은 COLMAP 그대로, global estimation만 교체).

## 한 줄 요약

> incremental이 이겨온 이유는 global의 **translation averaging이 원리적으로 불량(스케일 모호·intrinsics 의존·콜리니어 퇴화)**해서라는 진단 아래, 그 단계를 통째로 건너뛰고 **회전 고정 후 카메라 위치와 3D 포인트를 단일 목적함수로 동시 추정**(BATA류 각도 오차, 무작위 초기화에서 수렴)하는 global positioning을 넣어 — ETH3D SLAM에서 COLMAP 대비 recall +8%p·AUC@0.1m +9점을 내면서, LaMAR에서 **수십 배(HGE 기준 약 20배)**, IMC 2023에서 **약 8배** 빠른 범용 global SfM.

## 문제 — global SfM은 왜 정확도에서 밀렸나

두 패러다임은 대응 검색까지는 같고 그 뒤가 갈린다.

| | incremental (COLMAP) | global (기존 OpenMVG·Theia) |
|---|---|---|
| 골격 | 2-view 시드 → 이미지 등록·삼각측량·BA 반복 | rotation averaging → translation averaging → global triangulation → global BA |
| 정확도·강건성 | 높음 (매 단계 검증) | 낮음 — 격차의 주범이 이 논문의 표적 |
| 확장성 | 반복 BA 비용으로 제한 | 한 자릿수 배 이상 빠름 |
| 순차 데이터 | 순서 의존·drift 누적 | 콜리니어 모션에서 퇴화 |

원문(§1)이 정확도 격차의 주범으로 지목하는 것은 **translation averaging** — 상대 회전을 벗겨낸 뒤 pairwise 상대 translation 방향 $t_{ij}=\frac{c_j-c_i}{\|c_j-c_i\|}$만으로 전역 카메라 위치 $c_i$를 푸는 단계 — 이고, 실패 원인은 세 가지다.

1. **스케일 모호성**: 2-view 기하의 상대 translation은 방향만 있고 크기가 없다. 위치를 정하려면 triplet의 상대 방향들이 필요한데, triplet이 skewed triangle을 이루면 스케일 추정이 노이즈에 극도로 민감해진다.
2. **intrinsics 의존**: 상대 기하를 회전/translation으로 정확히 분해하려면 정확한 카메라 내부 파라미터가 필요하다. 인터넷 사진처럼 intrinsics가 부정확하면 translation 방향부터 크게 틀린다.
3. **콜리니어 퇴화**: 카메라들이 거의 일직선으로 움직이면(자율주행·핸드헬드 영상의 forward motion 등 순차 데이터에서 흔함) 문제가 퇴화한다. parallel rigidity 이론이 요구하는 view graph 조건이 깨지는 지점이다.

원문 §2.2의 관찰: 최근 연구들이 공통적으로 **이미지 포인트(3D structure)를 정식화에 끌어들일 때 translation averaging이 좋아졌다** — Wilson et al.의 "3D 포인트는 카메라 센터와 같은 방식으로 취급해 최적화에 편입 가능", Cui et al.의 point track 선형 제약, LiGT의 pose-only 선형 전역 제약 등. 공통 주제는 "3D 구조 제약이 카메라 위치 추정의 강건성·정확도를 돕는다"이고, GLOMAP은 이를 끝까지 밀어붙여 — translation averaging을 개선하는 게 아니라 **생략**한다. 별도로, global triangulation 쪽도 문제였다: DLT·midpoint류 다중 뷰 삼각측량은 임의 수준의 아웃라이어 앞에서 자주 깨지고, COLMAP의 RANSAC 기반 track 분할이 그 대응책이었다. GLOMAP은 삼각측량 역시 별도 단계로 두지 않고 카메라 위치와 **한 최적화 안에서** 푼다.

## 방법

파이프라인(Fig. 2): 대응 검색(특징 추출→매칭→2-view 추정→view graph calibration→상대 포즈 분해) → **global estimation**(rotation averaging → global positioning → global BA → structure refinement[선택]).

**COLMAP에서 재사용하는 것** — 이 논문은 앞단을 발명하지 않는다:

- 대응 검색은 COLMAP 구현 그대로: RootSIFT 특징 + scalable bag-of-words 이미지 검색으로 겹칠 후보 쌍을 찾고 brute-force 매칭.
- 2-view 기하 검증: F(미보정)/E(보정)/H(평면·순수 회전)를 추정해 기하적으로 불가능한 매칭 제거, intrinsics를 대략 알면 $R_{ij}\in SO(3)$, $t_{ij}\in\mathbb{R}^3$로 분해. 이렇게 얻은 2-view 기하가 view graph $\mathcal{G}$를 이룬다.
- view graph calibration은 Sweeney et al. 방식으로 기하 검증된 쌍에서 intrinsics를 갱신한 뒤, 갱신된 intrinsics로 상대 포즈를 다시 추정한다.

### 1) Feature track construction (§3.1)

2-view 기하 검증의 inlier 대응만 쓰되, 초기 분류를 존중한다: H가 최적 모델이면 H로 inlier를 검증하고, E/F도 동일 원칙. 추가로 **cheirality test**로 아웃라이어를 거르고, **epipole에 가까운 매칭과 삼각측량 각도가 작은 매칭은 불확실성이 커서 제거**한다. 남은 매칭을 view graph 전체에서 연결(concatenate)해 feature track을 만든다.

### 2) Rotation averaging (§2.2, §3.5)

표준 정식화 $\arg\min_{R}\sum_{i,j}\rho\left(d(R_j^\top R_{ij}R_i,\,I)^p\right)$에 대해 Chatterjee & Govindu(ICCV 2013)의 자체 구현을 쓴다 — 대규모에서 확장 가능하고 노이즈·아웃라이어 회전에 강건하다는 것이 채택 이유. 이후 $R_{ij}$와 $R_jR_i^\top$ 사이 **각도 거리를 임계값으로 잘라 비일관 상대 포즈를 view graph에서 제거**한다. 새 정식화를 제안하기보다 검증된 방법을 시스템에 맞게 조인 것이고, 논문 스스로 rotation averaging 실패(회전 대칭 구조)를 주요 실패 모드로 남겨둔다.

### 3) Global positioning — 핵심 기여 (§3.2)

translation averaging + global triangulation 대신 **카메라 위치 $c_i$와 3D 포인트 $X_k$를 한 번에** 푼다. 재투영 오차는 다중 뷰에서 highly non-convex라 초기화가 어렵고 오차가 unbounded라 아웃라이어에 약하다 — 그래서 BATA(Zhuang et al., CVPR 2018)의 정규화된 방향 차이 오차를 가져오되, **원래 BATA의 상대 translation 제약은 버리고 camera ray 제약만 남긴다**:

$$\arg\min_{X,\,c,\,d}\sum_{i,k}\rho\left(\left\|v_{ik}-d_{ik}(X_k-c_i)\right\|_2\right),\quad \text{s.t. } d_{ik}\ge 0$$

여기서 $v_{ik}$는 카메라 $c_i$에서 포인트 $X_k$를 보는 **전역 회전된 camera ray**(rotation averaging 결과로 회전만 미리 세계좌표계로 돌려놓은 관측 방향), $d_{ik}$는 정규화 인자다. Fig. 3의 그림이 직관을 준다: 회전(방향)은 고정한 채, 실측 ray(실선)와 포인트-카메라를 잇는 ray(점선) 사이의 **각도**가 줄도록 카메라 위치와 포인트 위치를 함께 움직인다. 최적 $d_{ik}$에서 이 오차는 ray와 $X_k-c_i$ 사이 각 $\theta$에 대해

$$\begin{cases}\sin\theta & \theta\in[0,\pi/2)\\ 1 & \theta\in[\pi/2,\pi]\end{cases}$$

와 동치 — **오차가 $[0,1]$로 유계**라 아웃라이어가 해를 끌고 가지 못한다. robustifier는 Huber, 최적화는 Ceres의 Levenberg–Marquardt. 결정적 디테일: **모든 포인트·카메라 변수를 $[-1,1]$ uniform random으로 초기화**하고 $d_{ik}=1$로 시작해도 bilinear 구조 덕에 안정적으로 수렴한다(보충 S5: 완벽한 관측+0px 노이즈에서 무작위 초기화로 GT에 신뢰성 있게 수렴, 64px 노이즈까지 AUC가 완만히 감소). intrinsics 미상 카메라의 항은 가중치 1/2로 낮춘다.

상대 translation 항을 버린 것의 실익 두 가지(§3.2 말미): (i) 상대 translation 추출에는 정확한 intrinsics가 필수인데 이를 안 쓰므로 **인터넷 사진·미보정 카메라에서도 동작**하고, 나쁜 intrinsics의 피해가 해당 카메라에 국한된다(다른 카메라로 전파 안 됨). (ii) feature track은 pairwise 방향과 달리 여러 카메라를 동시에 구속하므로 **콜리니어(forward/sideward) 모션에서 퇴화하지 않는다**.

### 4) Global BA와 정제 (§3.3, §3.5)

global positioning 결과는 강건하지만 intrinsics 미상이면 정확도가 부족 → 표준 재투영 오차 $\arg\min_{\pi,P,X}\sum_{i,k}\rho(\|\pi_i(P_i,X_k)-x_{ik}\|_2)$를 LM+Huber로 수 라운드 최적화한다. 디테일:

- 각 라운드에서 **카메라 회전을 먼저 고정하고 intrinsics+포인트만 최적화한 뒤 전체를 푼다** — 순차 데이터 재구성에 특히 중요한 설계.
- 첫 BA 전에 각도 오차 기반 3D 관측 pre-filtering(미보정 카메라는 임계 완화), 이후 이미지 공간 재투영 오차로 track 필터링.
- **필터된 track 비율이 0.1% 아래로 떨어지면 반복 종료**.
- 선택적 structure refinement: 추정 포즈로 재삼각측량(track 완성도 보강, COLMAP 방식) + global BA 라운드 반복 — 정확도를 더 끌어올리는 옵션.

### 5) Camera clustering (§3.4)

인터넷 컬렉션에서 무관한 이미지가 잘못 매칭돼 서로 다른 재구성이 하나로 붕괴하는 것을 후처리로 분리한다.

- 이미지 쌍별 공가시 포인트 수로 covisibility graph $\mathcal{G}$ 구성. 5개 미만 쌍은 상대 포즈를 신뢰할 수 없어 폐기하고, 남은 쌍 count의 **중앙값으로 inlier 임계 $\tau$** 설정.
- $\tau$ 초과 엣지만으로 강결합 성분(well-constrained cluster) 탐색.
- 두 성분 사이 $0.75\tau$ 초과 엣지가 2개 이상이면 병합 — 더 이상 병합이 없을 때까지 재귀 반복. 성분별로 별도 재구성 출력(보충 S4/Fig. S3: 떠 있는 위성 구조 제거 효과 확인).

## 결과 — 원문 수치

비교 대상은 OpenMVG·Theia(global), COLMAP(incremental). 공정 비교를 위해 모든 방법에 동일한 feature match를 입력하고 대응 검색 시간은 런타임에서 제외(OpenMVG/Theia 자체 대응 검색도 시도했으나 COLMAP 것이 일관되게 우수). GLOMAP은 고정 설정 하나로 전 데이터셋 실행. 지표는 무순서 데이터에서 모든 이미지 쌍 간 상대 회전·translation 오차 최대값 기반 AUC, 순차 데이터에서는 robust RANSAC 정렬 후 카메라 위치 오차의 recall/AUC(콜리니어에서 상대 오차가 scale drift를 못 잡기 때문).

- **ETH3D SLAM** (순차·보정·mm급 GT, Table 1 평균): Recall@0.1m **66.4** vs COLMAP 57.9·Theia 62.8·OpenMVG 48.2. AUC@0.1m **57.0** vs 47.6/46.0/34.9, AUC@0.5m **65.7** vs 57.9/61.1/48.6. 러닝타임 **133.5s vs COLMAP 1115.4s** — 본문 표현으로 recall 약 8% 상회, AUC 9점(0.1m)·8점(0.5m) 추가, COLMAP은 한 자릿수 배(one order of magnitude) 느림. global 대비로는 recall +18%/+4%, AUC@0.1m 약 11점 상회. 순차 콜리니어 퇴화 주장의 직접 검증 무대(*planar* 100.0, *sfm* 97.0 등에서 격차 뚜렷).
- **ETH3D MVS rig** (Table 2 평균): AUC@1° **57.6** vs COLMAP 45.5(1개 씬 실패)·Theia 40.4·OpenMVG 0.2, AUC@5° **87.2** vs 69.1. 시간 793.8s vs 2857.0s — COLMAP보다 약 3.5배 빠름. 전 씬 재구성 성공.
- **ETH3D MVS DSLR** (무순서·소규모, Table 3 평균): AUC@1° **80.8** vs COLMAP 79.2 — 대등. 단 *exhibition_hall*은 회전 대칭 구조로 rotation averaging이 붕괴해 열세.
- **LaMAR** (수만 장 AR 벤치마크, Table 4 평균): Recall@1m **49.1** vs COLMAP 32.0·Theia 11.0, AUC@5m **50.9** vs 39.4. 시간 **12405.3s vs COLMAP 354660.4s**(HGE 단독으로는 12587s vs 249771s ≈ 20배). LIN은 메모리 한계로 structure refinement 생략. CAB은 모든 방법이 부진(주야 조명 변화·층간 대칭·반복 파사드).
- **IMC 2023** (미보정 인터넷 이미지, Table 5 평균): AUC@3° **69.6** vs COLMAP 65.3·Theia 42.4·OpenMVG 20.4 — global 대비 수 배, COLMAP 대비 3°/5°/10° 모두 약 4점 상회하며 **약 8배 빠름**(497.3s vs 4051.0s). 단 이 GT 자체가 COLMAP 기반이라 COLMAP에 유리한 조건임을 감안.
- **MIP360** (Table 6 평균): AUC@3° **97.5** vs COLMAP 96.5, 시간 221.0s vs 345.7s(1.5배 이상 빠름). 하류 NVS 검증(보충 S4, Instant-NGP): PSNR **26.82** vs COLMAP 26.71·Theia 22.68, SSIM 0.750 vs 0.749.
- **Ablation** (Table 7): global positioning에 상대 translation 제약을 도로 넣으면(BATA cam: ETH3D DSLR AUC@1° 73.5, cam+pt: 80.1) **포인트만 쓴 제안 방식(80.8)보다 나빠진다** — "상대 translation 항이 수렴과 성능을 해친다"는 설계 논거의 직접 증거. Theia의 LUD로 교체 시 77.2. IMC 2023에서도 같은 순서(LUD 64.3 < cam 62.1 < cam+pt 68.6 < pt 69.6, AUC@3°).
- **보충 실험**: HSfM(hybrid)은 intrinsics가 정확한 ETH3D rig/DSLR에서는 GLOMAP에 근접하지만, 순차·희소한 ETH3D SLAM/LaMAR에서 실패하고 intrinsics 미상(IMC/MIP360)에서 크게 뒤처진다. LiGT는 전반 열세(ETH3D SLAM Recall@0.1m 34.0). Strecha 데이터셋 카메라 위치 오차(mm)에서는 LiGT와 접전(GLOMAP이 Fountain-P11 2.79mm, Castle-P30 22.36mm 우세).

## 내 실습 연결

리콘랩스 프로덕션이 COLMAP 포즈 → 3DGS 학습 구조라, 이 논문은 "전처리기를 갈아끼우면 뭐가 달라지나"라는 실무 질문 그 자체다. COLMAP 생태계 안의 후속(같은 DB·같은 대응 검색)이라 파이프라인 교체 지점이 mapper 한 단계로 국한되는 것도 실무적으로 크다.

첫째 **처리량**: 캡처당 SfM 시간이 수 배~수십 배 줄면 배치 처리 병목이 GS 학습 쪽으로 이동하고, 재촬영 후 재처리 같은 반복 사이클이 실질적으로 싸진다. MIP360류 object-centric 캡처에서 AUC@3° 97.5 vs 96.5로 품질 손실 없이 1.5배+ 단축, 그리고 보충 S4의 Instant-NGP NVS에서 PSNR 26.82 vs 26.71로 하류 품질까지 동급이라는 수치는 GS 전처리 전환의 직접 근거다. 대규모 공간 캡처(매장·쇼룸급)로 가면 LaMAR의 수십 배 격차가 말하듯 전환 이득이 오히려 커진다.

둘째 **실패 모드의 교체**: incremental의 실패(시드 실패, 순차 등록 중 drift·모델 분열)는 사라지는 대신, rotation averaging 실패(회전 대칭 구조 — *exhibition_hall* 사례가 정확히 대칭 제품 캡처 실패와 같은 유형)와 잘못 병합된 재구성(camera clustering이 커버)이 새 관리 대상이 된다. 운영 관점에서는 "어느 단계에서 죽었나"의 진단 지점이 등록 루프 로그에서 averaging/positioning 수렴 로그로 바뀌는 셈이고, 실패 케이스 관리 체계를 그대로 옮길 수 없다는 뜻이다. 매칭이 무너지는 저텍스처·반복 패턴 실패는 앞단이 COLMAP 그대로라 동일하게 남는다.

셋째, global positioning의 유계 각도 오차 + 무작위 초기화 수렴이라는 성질은 석사 때의 기하(다중 뷰 제약, 삼각측량 퇴화 조건) 감각과 직결되고, robotics 쪽으로는 이 구조가 pose graph optimization/SLAM 백엔드와 같은 언어(회전-병진 분리, robust kernel, gauge freedom)다. 특히 "콜리니어 forward motion에서 translation averaging이 퇴화한다"는 분석은 자율주행·모바일 로봇 시퀀스가 바로 그 케이스라, robot perception 면접에서 COLMAP 리뷰와 짝지어 incremental vs global의 트레이드오프를 실무+이론 양쪽에서 설명하는 축으로 쓸 수 있다.

## 한 줄 평 / 한계

**한 줄 평.** 새 오차 함수 발명이 아니라 배치의 혁신 — BATA의 각도 오차에서 정작 translation averaging을 들어내고 포인트-카메라 ray 제약만 남긴 한 수로, "global은 빠르지만 부정확"이라는 20년 통념을 시스템 수준에서 뒤집었다. COLMAP 저자가 직접 COLMAP 생태계 안에 후속을 배포했다는 점에서 실무 전환 비용도 낮다.

**한계.** (1) rotation averaging이 남은 단일 실패점 — 회전 대칭·반복 구조에서 붕괴하며(원문이 Doppelgangers 결합을 후속 방향으로 언급), 순차 콜리니어는 풀었지만 대칭은 못 풀었다. (2) 대응 검색은 여전히 고전 파이프라인 의존 — 2-view 기하가 틀리거나 매칭 자체가 안 되면(급격한 외관·시점 변화) 열화 또는 파국적 실패라고 원문이 명시. (3) 포인트를 변수로 들고 가는 대가로 메모리가 커진다(LaMAR LIN에서 structure refinement 생략). (4) 정확도 상한은 결국 global BA가 정하므로, 초대규모에서 BA 비용은 공유하는 숙제다.
