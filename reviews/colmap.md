# COLMAP — Structure-from-Motion Revisited (CVPR 2016)

*10년째 모두가 쓰는 SfM의 기준선 — incremental SfM의 각 단계를 전부 다시 조인 논문*

Schönberger, Frahm, *Structure-from-Motion Revisited*, CVPR 2016. [공식 PDF](https://demuc.de/papers/schoenberger2016sfm.pdf) · [colmap.github.io](https://colmap.github.io) · [GitHub](https://github.com/colmap/colmap)

## 한 줄 요약

> incremental SfM 파이프라인의 다섯 지점 — scene graph 기하 검증, next best view 선택, 삼각측량, BA/재삼각측량, BA 파라미터화 — 을 각각 다시 설계해, Bundler·VisualSFM 대비 **더 많은 이미지를 등록하고 더 긴 track을 만들면서 Bundler보다 50배 이상 빠른** 범용 SfM 시스템(COLMAP)을 오픈소스로 내놓은 논문.

## 문제

무순서 인터넷 포토 컬렉션에서의 incremental SfM은 이미 성숙해 보였지만, 저자들이 §3에서 지적하는 실패는 구체적이다.

- **불완전 등록**: 경험적으로 등록 가능해야 할 이미지의 상당 부분을 시스템이 등록하지 못한다. 근사 매칭이 만든 불완전한 scene graph가 원인 중 하나.
- **깨진 모델과 drift**: 잘못된 등록·부정확한 scene structure가 누적되며 모델이 갈라지거나 표류한다. 등록과 삼각측량은 공생 관계다 — 이미지는 기존 구조에만 등록되고, 구조는 등록된 이미지에서만 삼각측량된다. 매 단계에서 둘 다의 정확도·완전성을 최대화해야 하는데 기존 시스템은 여기서 샌다.
- **robustness·accuracy·completeness·scalability**를 동시에 만족하는 "진짜 범용" SfM은 아직 없다는 것이 논문의 출발점이다.

## 방법

파이프라인 골격은 표준 incremental SfM이다: 대응 검색(특징 추출 → 매칭 → 기하 검증) → scene graph → 2-view 초기화 → 이미지 등록(PnP+RANSAC) → 삼각측량 → BA → 아웃라이어 필터링의 반복. 기여는 이 골격의 각 마디를 갈아끼운 것이다(§4).

### 1) Scene graph augmentation — multi-model 기하 검증

이미지 쌍마다 하나의 모델(F)만 맞추는 대신 **H/E/F를 함께 추정해 쌍의 기하 유형을 라벨링**한다.

- F의 inlier 수 $N_F$가 임계 이상이면 검증 통과. 같은 쌍에서 homography inlier $N_H$를 세어 $N_H/N_F < \epsilon_{HF}$면 일반(general) 장면, 아니면 평면/회전 의심 — GRIC류 모델 선택의 근사다.
- 보정된 카메라라면 E도 추정해 $N_E/N_F > \epsilon_{EF}$로 캘리브레이션의 정합성을 확인하고, E를 분해·삼각측량해 얻은 **중위 삼각측량 각도 $\alpha_m$으로 순수 회전(파노라마)과 평면 장면을 구분**한다.
- 인터넷 사진 특유의 워터마크·타임스탬프·프레임(WTF)은 이미지 경계부에서의 similarity transform inlier $N_S$로 검출해($N_S/N_F$ 또는 $N_S/N_E$가 임계 초과) scene graph에서 제외한다.
- 이렇게 라벨된 그래프로 **초기화는 파노라마가 아닌, 가급적 보정된 쌍에서만 시드**하고, 파노라마 쌍에서는 삼각측량 자체를 하지 않아 퇴화 포인트를 차단한다.

### 2) Next best view selection — 멀티 해상도 격자 스코어

다음에 등록할 이미지 선택은 한 번의 오판이 mis-registration의 연쇄로 이어지는 지점. 후보는 삼각측량된 포인트를 $N_t>0$개 보는 미등록 이미지들이고, 불확실성 전파(Haner 등) 기반 정확한 방법은 후보마다 공분산을 계산해야 해 인터넷 규모에선 불가능하다. 제안은 **보이는 포인트의 수와 분포를 동시에 근사하는 격자 스코어**다.

- 이미지를 $K_l \times K_l$ 격자로 이산화하고, 셀이 처음 채워질 때만 가중치 $w_l$을 스코어 $\mathcal{S}$에 더한다 — 같은 수의 포인트라면 한 구석에 몰린 것보다 고르게 퍼진 쪽이 높은 점수.
- 포인트가 적을 때의 분포 분해능을 위해 $l = 1...L$ 레벨 피라미드로 확장: 각 레벨 해상도 $K_l = 2^l$, 가중치

$$w_l = K_l^2$$

- 로 전 레벨 점수를 누적한다. 온라인으로 효율적으로 갱신 가능하며, 관측이 많고 고르게 분포한(=PnP가 잘 조건화된) 이미지가 먼저 등록된다.

### 3) Robust triangulation — RANSAC 다중 뷰 삼각측량

feature track은 epipolar line 위의 애매한 매칭 때문에 아웃라이어가 많고, 서로 다른 3D 포인트의 track이 하나로 잘못 합쳐지기도 한다(길이 같은 track 4개가 합쳐지면 아웃라이어 비율 75%). Bundler는 모든 쌍 조합을 시도한 뒤 전체 track으로 multi-view 삼각측량을 하는데, 아웃라이어에 취약하고 비싸다. 제안은 **track $\mathcal{T} = \{T_n\}$ 위의 RANSAC**이다.

- 두 관측을 샘플해 DLT로 삼각측량하고, 그 2-view 해가 well-conditioned인지 두 조건으로 본다: 충분한 삼각측량 각도

$$\cos\alpha = \frac{\mathbf{t}_a - \mathbf{X}_{ab}}{\lVert \mathbf{t}_a - \mathbf{X}_{ab} \rVert_2} \cdot \frac{\mathbf{t}_b - \mathbf{X}_{ab}}{\lVert \mathbf{t}_b - \mathbf{X}_{ab} \rVert_2}$$

- 와 두 뷰 모두에서의 양의 깊이(**cheirality**). track의 다른 관측 $T_n$은 깊이 $d_n > 0$이고 재투영 오차 $e_n$이 임계 $t$ 미만이면 inlier.
- 작은 track에서 같은 최소 집합을 중복 샘플하지 않도록 유일 샘플러를 쓰고, 신뢰도 $\eta$ 기반 적응형 반복(초기 inlier 비율 $\epsilon_0$에서 시작해 consensus가 커질 때마다 $K$ 갱신).
- 하나의 track에 여러 독립 포인트가 숨어 있을 수 있으므로 **consensus set을 제거하고 재귀적으로 반복**, 최신 consensus 크기가 3 미만이면 정지 — 잘못 병합된 track에서 복수의 포인트를 회수한다.

### 4) BA 전략 — local/global BA + 재삼각측량 + 반복 정제

- **Local BA**: 등록 직후에는 모델이 국소적으로만 변하므로, 새 이미지와 가장 연결이 강한 이미지 집합에 대해서만 BA. 손실은 아웃라이어를 눌러 주는 Cauchy $\rho_j$.
- **Global BA**: VisualSFM처럼 모델이 일정 비율 이상 성장했을 때만 수행해 상각 선형 런타임. 수백 대까지는 sparse direct solver, 그 이상은 PCG(Ceres 사용). 인터넷 사진은 radial distortion 1개짜리 단순 카메라 모델로 순수 자가보정. principal point는 ill-posed라 중심에 고정.
- **필터링**: BA 후 재투영 오차 큰 관측 제거, 포인트마다 시선쌍 최소 삼각측량 각도 검사, global BA 후 비정상 FOV·왜곡 계수의 퇴화 카메라(파노라마·편집 사진) 제거.
- **재삼각측량(RT)**: VisualSFM의 pre-BA RT에 더해 **post-BA RT를 새로 추가** — BA로 좋아진 포즈를 이용해, 임계 이하 오차의 관측으로 이전에 실패한 track을 이어 붙이고 track 병합도 시도.
- **반복 정제**: Bundler·VisualSFM은 BA·필터링을 1회만 하지만, drift 때문에 첫 BA는 아웃라이어에 오염돼 있다. **BA → RT → 필터링을 필터링되는 관측 수가 줄어들 때까지 반복**하며, 대부분 2회째에 크게 좋아지고 수렴한다.

### 5) Redundant view mining — 겹치는 카메라의 그룹 파라미터화

인터넷 컬렉션은 인기 시점에 카메라가 뭉쳐 있는 비균일 분포다. 최근 확장의 영향을 받지 않은 장면 부분을 고겹침 이미지 그룹 $G_r$들로 분할하고, **그룹 내 카메라들을 그룹-로컬 좌표계의 단일 카메라로 접어** BA 비용을 줄인다. 그룹화 척도는 이진 가시성 벡터의 비트 연산

$$V_{ab} = \lVert \mathbf{v}_a \wedge \mathbf{v}_b \rVert \, / \, \lVert \mathbf{v}_a \vee \mathbf{v}_b \rVert$$

로, $V_{ab} > V$면 같은 그룹(최대 크기 $S$, 시선 방향 $\pm\beta$ 이내의 공간 최근접 $K_r$개만 탐색). 그룹 이미지의 비용은 그룹 외부 파라미터 $\mathbf{G}_r$과 고정된 $\mathbf{P}_c$의 합성 $\mathbf{P}_{cr} = \mathbf{P}_c \mathbf{G}_r$로 투영하는 $E_g$가 된다. 최근 확장에 영향받은 이미지(새로 추가됐거나 관측의 $\epsilon_r$ 비율 이상이 $r$px 초과 오차)는 각자 크기 1 그룹 = 표준 BA로 정밀 정제.

## 결과

17개 데이터셋, 총 144,953장의 무순서 인터넷 사진(+ ground-truth 포즈가 있는 Quad). RootSIFT + vocabulary tree로 100 최근접 이웃 매칭, 2.7GHz·256GB RAM 머신. 비교 대상은 incremental(Bundler, VisualSFM)과 global(DISCO, Theia).

- **등록 완전성(Table 1)**: Rome 74,394장 중 **20,918장 등록 — Bundler 13,455 · VisualSFM 14,797** 대비 압도적. Quad는 5,860 (VSFM 5,624, Bundler 5,028), Dubrovnik 5,913. 모든 데이터셋에서 타 시스템 대비 완전성 최고, 특히 큰 데이터셋에서 격차가 크다.
- **track 길이와 정확도**: Alamo에서 평균 track 길이 **11.6** (Bundler 4.5, VSFM 8.9) — 길어진 track이 BA의 중복성을 높인다. 평균 재투영 오차는 Alamo 0.68px (Bundler 2.29, VSFM 0.70)처럼 전반적으로 최저 수준.
- **속도**: Rome 기준 10,912s vs Bundler 295,200s — 본문 표현으로 **Bundler보다 50배 이상 빠르고**, VisualSFM보다 약간 느리며 Theia가 가장 빠르다.
- **포즈 정확도(Quad, ground-truth 카메라와의 중위 거리)**: **Ours 0.85m** < VisualSFM 0.89m < Bundler 1.01m < DISCO 1.16m.
- **삼각측량(Dubrovnik, 47M 매칭 → 2.9M track, Table 2)**: 재귀 RANSAC($\eta_2=0.5$)이 **906,501 포인트 / 평균 track 길이 8.795 / 7.82M 샘플** — Bundler(713,824 포인트 / 7.824 / 136.34M 샘플) 대비 포인트 27% 더 많고 샘플 수는 1/17. exhaustive 대비 **10–40배 빠르면서** track 품질은 근소 열세.
- **Redundant view mining**: 겹침 임계 $V$=0.6/0.3/0.1에서 전체 런타임 5%/14%/32% 단축, 평균 재투영 오차는 0.26px → 0.27/0.28/0.29px로 미세 악화. Colosseum은 $V$=0.4로 전체 파이프라인 36% 단축에 동등한 재구성.

## 내 실습 연결

리콘랩스의 제품 3D 캡처 파이프라인이 정확히 **COLMAP 포즈 → 3DGS 학습** 순서다. 현장에서 체득한 감각 — "COLMAP 포즈 품질이 GS 품질의 상한을 정한다" — 의 근거가 이 논문 곳곳에 있다: GS가 아무리 잘 학습돼도 등록 안 된 뷰는 복원에 기여하지 못하고(§3의 불완전 등록 문제), drift 낀 포즈는 float와 blur로 그대로 새어 나온다(반복 정제가 잡으려는 바로 그 오염). 프로덕션에서 겪는 대표 실패 모드인 **저텍스처·반복 패턴 제품에서의 등록 실패**는 이 파이프라인의 입구, 즉 대응 검색의 한계다 — COLMAP의 기하 검증·NBV·robust triangulation은 "대응이 있다면" 그 뒤를 최대한 살리는 장치지, 대응 자체를 만들어 주진 않는다. 석사 때 하던 대응점(stereo matching) 연구가 이 지점과 만난다: 매칭이 무너지는 조건을 아는 것이 SfM 실패를 예측·회피하는 능력이고, 캡처 가이드(뷰 간격, 텍스처 보강)로 되먹임된다. 또 NBV의 "관측 수 + 격자 분포" 스코어는 캡처 프로토콜 설계와 같은 언어다 — 어떤 각도의 사진을 추가해야 등록이 사는지에 대한 정량적 답. robotics 쪽으로는 이 파이프라인 전체가 SLAM의 오프라인 쌍둥이(PnP 등록 ↔ localization, RT+BA ↔ loop closure 후 최적화)라 robot perception 면접에서 가장 자신 있게 깊이를 보여줄 수 있는 축이다.

## 한 줄 평 / 한계

**한 줄 평.** 새 이론 하나로 승부하는 논문이 아니라, 파이프라인의 병목 다섯 곳을 각각 공학적으로 다시 조인 시스템 논문 — 그 결과물이 10년째 NeRF/3DGS 시대의 사실상 표준 전처리기로 살아남았다는 것 자체가 설계의 정당성 증명이다.

**한계.** (1) 대응 검색은 여전히 SIFT류 handcrafted 특징에 의존 — 저텍스처·반복 패턴·야간 등 매칭이 무너지는 곳에서는 뒷단이 아무리 강건해도 구제 불가이며, 이후 learned feature(SuperPoint/SuperGlue류)와 detector-free matcher가 이 입구를 대체해 온 이유다. (2) incremental 구조라 순차 등록의 순서 의존성과 누적 drift가 원리적으로 남고, global BA 비용 때문에 초대규모에서는 여전히 비싸다. (3) 각 모듈의 임계값($\epsilon_{HF}$, $\alpha$, $t$, $V$ 등)이 많아 도메인이 바뀌면(예: 근접 제품 캡처 vs 인터넷 랜드마크) 재튜닝 여지가 있다 — 프로덕션에서 파라미터 프로파일을 따로 관리하게 되는 이유이기도 하다.
