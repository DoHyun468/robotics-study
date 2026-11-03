# Motion Tracking — 시간축의 대응 문제: 고전 flow에서 학습 트래커까지

스테레오 매칭이 **공간축의 대응**(같은 순간, 다른 시점)이라면, 트래킹은 **시간축의 대응**(같은 대상, 다른 순간)이다. 재료도 같다 — 밝기/특징의 유사성과 기하·운동의 제약. 이 페이지는 그 시간축 대응을 1981년의 밝기 항등식부터 transformer 점 추적까지 계보로 정리한다. 칼만 필터의 유도와 다중 객체 추적(MOT)의 상세는 [av_stack](av_stack.md) §1·§5에 이미 있으므로 여기서는 연결만 하고, 이 페이지는 **"픽셀·점·객체를 프레임 사이에서 어떻게 잇는가"** 의 알고리즘 자체에 집중한다.

## 1. 문제 지도 — 무엇을 추적하는가

| 문제 | 출력 | 대표 계보 |
|---|---|---|
| 조밀 optical flow | 모든 픽셀의 (u, v) | Horn–Schunck → RAFT |
| 희소 점 추적 | 선택된 코너들의 궤적 | Lucas–Kanade → KLT |
| 단일 객체(SOT) | 한 객체의 박스/마스크 | MOSSE·KCF → Siamese → transformer |
| 다중 객체(MOT) | 검출들의 ID 연결 | SORT·DeepSORT·ByteTrack ([av_stack §5](av_stack.md)) |
| 장기 점 궤적(TAP) | 임의 점의 장기 추적 + 가림 플래그 | PIPs·TAPIR·CoTracker |

모든 트래커는 두 재료의 배합이다.

- **운동 모델**: 다음 프레임에 어디쯤 있을지의 사전 예측 — 칼만의 예측 단계, flow의 초기값, 탐색 윈도의 위치.
- **외관 모델**: 거기서 무엇이 "같은 것"인지의 판정 — 밝기 패치, 히스토그램, correlation filter, 학습된 임베딩.

고전사는 대체로 외관 모델의 진화사이고, 운동 모델은 등속(CV) 칼만에서 크게 바뀌지 않았다 — 바뀐 것은 "외관이 강해질수록 운동 모델 의존이 줄어드는" 배합비다.

## 2. 출발점 — 밝기 항등 가정과 optical flow 제약식

모든 고전 트래킹의 뿌리는 가정 하나다. **같은 물리적 점은 짧은 시간 동안 같은 밝기로 보인다**(brightness constancy):

$$I(x + u,\; y + v,\; t + 1) = I(x, y, t)$$

$(u, v)$가 그 픽셀의 모션(optical flow)이다. 움직임이 작다고 보고 좌변을 1차 Taylor 전개하면

$$I(x, y, t) + I_x u + I_y v + I_t \approx I(x, y, t)$$

양변에서 $I(x,y,t)$가 지워지고 **optical flow 제약식**이 남는다:

$$I_x u + I_y v + I_t = 0 \qquad \Longleftrightarrow \qquad \nabla I \cdot (u, v)^T = -I_t$$

여기서 $I_x, I_y$는 공간 그라디언트, $I_t$는 시간 차분 — 전부 이미지에서 **잴 수 있는 양**이다. 그런데 식이 픽셀당 1개, 미지수는 2개다. 이 부족이 **aperture problem**이다: 이 식은 flow의 **그라디언트 방향 성분만** 결정하고, 등밝기선(edge) 방향 성분은 아무 값이나 가능하다. 작은 구멍으로 움직이는 빗금을 보면 실제 운동과 무관하게 빗금의 법선 방향으로만 흐르는 듯 보이는 현상 — 창문(aperture) 안의 정보만으로는 운동이 유일하게 정해지지 않는다.

부족한 1개의 식을 어디서 가져오느냐로 두 고전이 갈린다. **이웃 픽셀에서 빌리면 Lucas–Kanade, 전역 매끄러움에서 빌리면 Horn–Schunck.** 1981년 같은 해에 나온 두 답이다.

## 3. Lucas–Kanade — 국소 창의 최소제곱

### 3.1 유도

가정: 작은 창 $W$(예: 21×21) 안의 픽셀들은 **같은 flow** $(u, v)$를 가진다. 창 안 픽셀 $n$개 각각이 제약식을 하나씩 주므로:

$$\underbrace{\begin{pmatrix} I_x(\mathbf{p}_1) & I_y(\mathbf{p}_1) \\ \vdots & \vdots \\ I_x(\mathbf{p}_n) & I_y(\mathbf{p}_n) \end{pmatrix}}_{A \in \mathbb{R}^{n \times 2}} \begin{pmatrix} u \\ v \end{pmatrix} = \underbrace{\begin{pmatrix} -I_t(\mathbf{p}_1) \\ \vdots \\ -I_t(\mathbf{p}_n) \end{pmatrix}}_{\mathbf{b}}$$

식 $n$개 > 미지수 2개의 과결정계 — 최소제곱으로 풀면 정규방정식:

$$A^T A \begin{pmatrix} u \\ v \end{pmatrix} = A^T \mathbf{b}, \qquad A^T A = \begin{pmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{pmatrix}$$

이 2×2 행렬 $G = A^T A$가 **structure tensor**다. 해가 안정적으로 존재할 조건은 $G$가 잘 조건화되는 것, 즉 **두 고유값이 모두 충분히 큰 것**이다. 고유값의 의미를 읽으면:

- $\lambda_1 \approx \lambda_2 \approx 0$: 창 안이 균일(무텍스처) — 그라디언트가 없어 아무것도 못 정함
- $\lambda_1 \gg \lambda_2 \approx 0$: 창 안이 한 방향 edge — aperture problem이 창 수준에서 재발
- $\lambda_1, \lambda_2$ 둘 다 큼: 여러 방향의 그라디언트가 공존 = **코너** — flow가 유일 결정

### 3.2 Shi–Tomasi — "추적 가능성"이 코너 검출을 낳다

위 조건을 뒤집으면 검출기가 된다. **Good Features to Track**(Shi & Tomasi, 1994)의 기준

$$\min(\lambda_1, \lambda_2) > \tau$$

는 "코너라서 뽑는" 것이 아니라 **"LK 방정식이 풀리는 곳이라서 뽑는"** 것이다 — 코너 검출의 유명한 기준이 사실 트래킹의 가해성(solvability) 조건에서 나왔다는 것이 이 계보의 묘미다. Harris 응답 $\det G - k(\operatorname{tr} G)^2$는 같은 $G$를 다른 스칼라로 요약한 사촌이다.

### 3.3 반복·피라미드 — 실전 KLT

Taylor 전개는 움직임이 1~2픽셀일 때만 유효하다. 실전은 두 겹의 보강으로 해결한다.

**반복(Gauss–Newton)**: 1차 해 $(u, v)$로 패치를 워핑해 놓고 남은 잔차로 다시 풀기를 수렴까지 반복. 사실 LK는 비선형 최소제곱

$$\min_{\mathbf{d}} \sum_{\mathbf{p} \in W} \big[ I_1(\mathbf{p} + \mathbf{d}) - I_0(\mathbf{p}) \big]^2$$

의 Gauss–Newton이고, $G = A^T A$는 정확히 [캘리브레이션에서 본](vision/calibration.md) $J^T J$ 헤시안 근사다 — LM의 2×2 미니어처.

**피라미드**: 이미지를 1/2씩 줄인 피라미드의 꼭대기(움직임이 1/2^L로 축소된 곳)에서 풀고, 그 해를 다음 레벨의 초기값으로 전파. 수십 픽셀의 움직임도 각 레벨에서는 "작은 움직임"이 된다.

여기에 창의 변형까지 허용(평행이동 → affine 6파라미터)하고, 누적 drift를 감시하기 위해 첫 프레임 템플릿과의 affine 정합을 주기적으로 검사하는 것이 **KLT 트래커**(Kanade–Lucas–Tomasi)의 완성형이다. 실무 위생 수칙 하나: 순방향으로 추적한 점을 역방향으로 되추적해 제자리로 돌아오는지 보는 **forward–backward 검사** — [MVS의 geometric consistency](reviews/colmap.md)와 같은 "왕복 합의" 철학이다.

## 4. Horn–Schunck — 전역 평활의 변분법

부족한 식을 "flow 장은 전체적으로 매끄럽다"는 사전 지식으로 채우는 반대편 답. 에너지 범함수

$$E(u, v) = \iint \underbrace{(I_x u + I_y v + I_t)^2}_{\text{데이터 항}} + \alpha^2 \underbrace{\big( \|\nabla u\|^2 + \|\nabla v\|^2 \big)}_{\text{평활 항}} \; dx\, dy$$

를 **모든 픽셀의 flow에 대해 동시에** 최소화한다. Euler–Lagrange 방정식을 세우면 라플라시안이 등장해 이웃과 결합되고, $\Delta u \approx \bar u - u$(이웃 평균과의 차) 근사로 Jacobi 반복해를 얻는다:

$$u^{k+1} = \bar u^k - \frac{I_x \big( I_x \bar u^k + I_y \bar v^k + I_t \big)}{\alpha^2 + I_x^2 + I_y^2}, \qquad v^{k+1} = \bar v^k - \frac{I_y \big( I_x \bar u^k + I_y \bar v^k + I_t \big)}{\alpha^2 + I_x^2 + I_y^2}$$

읽는 법: 각 픽셀은 **이웃 평균을 받아들이되, 자기 데이터 항이 허용하는 방향으로 수정**한다. 그 결과 그라디언트가 있는 곳의 정보가 무텍스처 영역으로 **확산**되어 조밀한 flow가 채워진다 — LK가 비워두는 곳을 HS는 메운다. 대가는 물체 경계: 평활 항이 경계를 모르는 채 문지르므로 서로 다른 운동이 번져 섞인다. 이후 30년의 변분 flow 연구(robust 페널티, TV-L1, 경계 보존 가중치)는 이 평활 항을 어떻게 똑똑하게 만드느냐의 역사였고, §9의 RAFT가 그 전통을 학습으로 승계한다.

| | Lucas–Kanade | Horn–Schunck |
|---|---|---|
| 부족한 식의 출처 | 국소 창의 공유 가정 | 전역 평활 정칙화 |
| 출력 | 희소 (풀리는 점만) | 조밀 (전 픽셀) |
| 무텍스처 | 포기 (Shi–Tomasi로 회피) | 이웃에서 확산해 채움 |
| 실패 모드 | aperture, 큰 움직임 | 운동 경계 번짐 |
| 계산 | 점당 2×2 — 실시간 | 전역 반복 — 무거움 |

## 5. 상태 공간 필터 — 운동 모델 담당 ([av_stack §1](av_stack.md))

외관 매칭이 "어디가 닮았나"를 답한다면, 칼만 필터는 **"어디쯤 있어야 하나"의 사전분포**를 공급한다 — 탐색 윈도의 중심을 정하고, 말이 안 되는 매칭을 게이팅으로 걸러주는 역할. 예측–갱신 2박자와 칼만 게인의 유도, 등속(CV) 운동 모델, EKF/UKF/파티클 필터로의 확장은 [av_stack §1](av_stack.md)에 전부 전개되어 있으므로 반복하지 않는다. 이 페이지의 관점에서 기억할 한 줄만: **필터는 외관 모델이 틀렸을 때를 버텨주는 관성이고, 외관 모델이 강해질수록(§7~§8) 그 역할은 게이팅으로 축소된다.**

## 6. 히스토그램 외관 — mean-shift 트래킹

박스 안 색 히스토그램 $q$(템플릿)와 후보 위치의 히스토그램 $p(\mathbf{y})$의 유사도를 Bhattacharyya 계수

$$\rho(\mathbf{y}) = \sum_u \sqrt{p_u(\mathbf{y})\, q_u}$$

로 재고, 이 유사도 지형을 **mean-shift**(커널 가중 평균으로 이동 벡터를 계산하는 경사 상승)로 올라가 모드를 찾는다. 공간 구조를 버리고 분포만 보므로 비강체 변형에 강하고 극도로 싸지만, 같은 이유로 배경과 색이 겹치면 무너진다. "외관을 어디까지 요약할 것인가"라는 스펙트럼에서 한쪽 끝(전부 요약)을 차지하는 고전 — 반대쪽 끝(공간 구조 전부 유지)이 다음 절의 correlation filter다.

## 7. Correlation Filter 계열 — 주파수 영역의 폐형식 학습

2010년대 전반 SOT를 지배한 계보. 핵심 발상: **추적을 "매 프레임 분류기를 다시 학습하는 문제"로 보되, 학습을 FFT로 폐형식화해 수백 fps로 만든다.**

### 7.1 MOSSE — ridge 회귀를 주파수 영역에서

목표: 입력 패치 $f$에 correlation을 취했을 때 원하는 응답 $g$(객체 중심에 봉우리를 둔 가우시안)가 나오는 필터 $h$를 찾기. correlation 정리로 주파수 영역에서는 요소별 곱이 되므로($\hat{\,}$ = DFT, $\odot$ = 요소별 곱, $^*$ = 켤레):

$$\min_{\hat h} \sum_i \big\| \hat f_i \odot \hat h^* - \hat g_i \big\|^2 \quad \Longrightarrow \quad \hat h^* = \frac{\sum_i \hat g_i \odot \hat f_i^*}{\sum_i \hat f_i \odot \hat f_i^* + \lambda}$$

**요소별 나눗셈 한 번이 학습의 전부다.** 새 프레임마다 분자·분모를 이동평균으로 갱신하면 온라인 적응까지 공짜. 2010년 MOSSE가 669fps를 찍으며 이 계보를 열었다.

### 7.2 KCF — 순환행렬과 커널의 결합

MOSSE의 수학적 정체를 밝힌 것이 KCF(2015)다. 패치 $\mathbf{x}$의 **모든 순환 시프트**를 행으로 쌓은 순환행렬 $X = C(\mathbf{x})$를 학습 데이터로 삼는 ridge 회귀를 생각하면 — 순환행렬은 DFT로 대각화된다는 사실

$$X = F \operatorname{diag}(\hat{\mathbf{x}})\, F^H$$

($F$ = DFT 행렬) 덕분에, 역행렬이 필요한 ridge 해가 요소별 연산으로 붕괴한다. 커널 버전(쌍대 변수 $\boldsymbol\alpha$)으로 쓰면:

$$\hat{\boldsymbol\alpha} = \frac{\hat{\mathbf{y}}}{\hat{\mathbf{k}}^{xx} + \lambda}, \qquad \text{응답 맵} \;\; \hat{\mathbf{f}}(\mathbf{z}) = \hat{\mathbf{k}}^{xz} \odot \hat{\boldsymbol\alpha}$$

$\mathbf{k}^{xx}$는 커널 자기상관(가우시안 커널도 FFT로 계산 가능). 해석: **"시프트된 패치 수천 장으로 학습한 커널 분류기"를 명시적 시프트 생성 없이 $O(n \log n)$에 학습·평가**한다. 시프트가 가상의 학습 샘플이 되는 것 — 데이터 증강을 대수 구조로 공짜 획득한 드문 사례다. 한계도 같은 구조에서 나온다: 순환 가정이 만드는 경계 아티팩트(패치 가장자리가 반대편과 이어진 것으로 취급됨), 고정 크기 창, 그리고 HOG 수준 특징의 표현력 — 이 셋이 다음 절의 학습 임베딩으로 넘어가는 동기가 된다.

## 8. 학습 기반 SOT — Siamese에서 one-stream transformer까지

### 8.1 SiamFC — correlation의 재료를 학습하다

구조는 correlation filter와 같은데 재료가 바뀐다. 템플릿 $z$(첫 프레임의 객체)와 탐색 영역 $x$를 **같은 CNN** $\varphi$에 통과시키고, 임베딩끼리 cross-correlation:

$$f(z, x) = \varphi(z) \star \varphi(x) + b$$

응답 맵의 봉우리가 객체 위치다. 학습은 대규모 비디오에서 "같은 객체 쌍은 봉우리, 아니면 바닥"이 되도록 $\varphi$를 훈련 — **온라인 학습을 아예 버리고, 오프라인에서 배운 유사도 함수 하나로 밀고 간다**는 전환이 포인트다(MOSSE·KCF의 온라인 적응과 정반대 베팅). SiamRPN은 응답 맵 위에 RPN 헤드를 얹어 박스의 크기·비율까지 회귀하고, SiamRPN++가 깊은 백본을 가능케 해 계보를 완성했다.

### 8.2 One-stream transformer — 추출과 비교의 경계가 사라지다

Siamese의 구조적 한계는 템플릿과 탐색 영역이 **각자 임베딩된 뒤에야** 만난다는 것 — 비교에 필요한 특징이 무엇인지 모르는 채 추출이 끝난다. OSTrack·MixFormer(2022) 계열은 템플릿 토큰과 탐색 토큰을 **처음부터 한 ViT에 섞어 self-attention**시킨다. 추출과 상호작용이 모든 레이어에서 동시에 일어나므로, 템플릿이 "지금 이 객체를 구분하는 데 중요한 특징"을 탐색 쪽 인코딩에 처음부터 주입한다. 현재 SOT 벤치마크의 표준 골격이고, SAM 2(2024)는 여기에 메모리 어텐션(과거 프레임들의 마스크·특징을 기억 은행으로)을 더해 "비디오 분할 = 마스크 트래킹"을 프롬프트 기반으로 통합했다.

| 세대 | 외관 모델 | 온라인 학습 | 대표 |
|---|---|---|---|
| 필터 | HOG + 주파수 ridge | 매 프레임 폐형식 | MOSSE, KCF |
| Siamese | 학습 임베딩 + correlation | 없음 (유사도만) | SiamFC, SiamRPN++ |
| one-stream | joint attention | 없음~경량 | OSTrack, MixFormer, SAM 2 |

## 9. RAFT — 조밀 flow의 현대 표준

Horn–Schunck 전통(조밀, 전역, 반복)의 학습판 완성형(ECCV 2020 최우수논문). 세 부품:

**① All-pairs correlation volume.** 두 프레임의 특징 맵 $f_1, f_2$(1/8 해상도)에서 **모든 픽셀 쌍**의 내적을 미리 계산한 4차원 볼륨

$$C_{ijkl} = \langle f_1(i, j),\; f_2(k, l) \rangle$$

을 만들고, 마지막 2차원을 평균 풀링한 피라미드 $\{C^1, C^2, C^4, C^8\}$로 보관한다 — "어디로든 움직일 수 있다"는 가능성 전체를 한 번에 테이블화한 것. 탐색 범위 제한(고전 flow의 작은 움직임 가정)이 구조에서 사라진다.

**② Iterative refinement GRU.** flow 추정 $\mathbf{f}^k$를 들고, 각 픽셀이 **자기 추정이 가리키는 곳 주변**의 correlation 값들을 볼륨에서 조회(lookup)해, GRU가 보정량을 출력한다:

$$\mathbf{f}^{k+1} = \mathbf{f}^k + \Delta \mathbf{f}^k, \qquad \Delta \mathbf{f}^k = \text{GRU}\big( \text{lookup}(C, \mathbf{f}^k),\; \mathbf{f}^k,\; h^k \big)$$

0에서 시작해 수십 회 반복하면 수렴한다. 이 구조는 우연이 아니다 — **LK/변분법의 "현재 추정 주변에서 잔차를 보고 업데이트"라는 최적화 반복을, 업데이트 규칙만 학습으로 바꾼 것**이다. 고전 최적화의 수렴 구조를 아키텍처에 박아 넣고(귀납 편향), 한 스텝의 똑똑함만 데이터에서 배우는 설계 — 모든 반복이 같은 가중치를 공유하므로 recurrent 최적화기로 읽힌다.

**③ Convex upsampling.** 1/8 해상도 flow를 각 고해상도 픽셀이 "주변 coarse 셀 9개의 학습된 볼록 결합"이 되도록 복원 — 경계가 bilinear보다 훨씬 날카롭다.

이후 계보는 한 줄씩: FlowFormer(볼륨을 transformer로 인코딩), SEA-RAFT(단순화·고속화), 그리고 GMFlow(상관을 attention 매칭으로 재정식화). 하지만 "4D 볼륨 + 반복 정제"라는 RAFT의 골격 자체는 그대로 살아 있고, [DROID-SLAM](slam.md)·[DUSt3R 계열](reviews/dust3r.md)의 대응 추정에도 이식됐다.

## 10. Point Tracking(TAP) — 장기 점 궤적이라는 새 문제

flow를 프레임마다 이어 붙여(chaining) 장기 궤적을 만들면 오차가 누적되고, 가림 한 번에 끊긴다. 2022년 전후로 문제가 재정의된다 — **"비디오의 아무 점이나 찍으면, 전체 길이에 걸쳐 그 점의 궤적과 가림 여부를 내라"**(Tracking Any Point).

- **PIPs**(2022): 한 점의 외관 특징 + 궤적을 시간 윈도 안에서 MLP-Mixer로 반복 정제 — RAFT의 "반복 정제"를 시간축으로 돌린 것. 가림 구간을 윈도 안에서 건너뛴다.
- **TAPIR**(2023): 전 프레임 글로벌 매칭으로 초기 궤적을 잡고(가림 뒤 재등장도 매칭으로 복구), PIPs식 반복 정제로 다듬는 2단 구조.
- **CoTracker**(2023): 여러 점의 궤적을 **트랙 간 attention**으로 함께 추정 — 같은 강체 위의 점들이 서로를 지지하므로, 가려진 점도 이웃 궤적이 "데리고 간다". 혼자 추적하던 문제를 집단 추적으로 바꾼 전환.

로보틱스에서 TAP가 주목받는 이유는 명확하다 — **시연 비디오의 물체 점 궤적이 그대로 "행동의 중간 표현"이 된다.** 사람 손 영상에서 물체가 어떻게 움직였는지를 점 궤적으로 뽑아 로봇 정책의 목표로 주는 계열(ATM, Track2Act, RoboTAP)이 이미 자리 잡았고, 이는 [VLA](vla.md)의 "언어 → 행동" 경로와 별개로 "궤적 → 행동"이라는 기하적 경로를 연다.

## 11. 다중 객체 추적(MOT) — 연결만 요약 ([av_stack §5](av_stack.md))

MOT는 "검출기가 매 프레임 내놓는 박스들을 ID로 잇는" 할당 문제로, 이 페이지의 단일 대상 기법들과 직교하는 층이다. 골격만 복기하면 — 트랙별 CV-칼만 예측, 예측 박스와 신규 검출의 비용 행렬(IoU 또는 마할라노비스 거리 + 외관 임베딩 코사인), 헝가리안 알고리즘 할당, 미연결 처리(신규/소멸). SORT(최소 구성) → DeepSORT(재식별 임베딩 추가) → ByteTrack(저신뢰 검출을 2차 연관에 재활용) 계보와 수식 전개는 [av_stack §5](av_stack.md)에 있다. 이 페이지의 기법들과의 관계 한 줄: **SOT·TAP이 강해질수록 MOT의 "검출 사이를 잇는" 부담이 줄어든다** — 가림 구간을 외관·궤적 모델이 버텨주기 때문이다.

## 12. 로보틱스 접속 — 트래킹은 어디에 꽂히는가

**SLAM front-end.** [VINS 계열](slam.md)의 front-end가 정확히 §3의 피라미드 KLT다 — Shi–Tomasi로 뽑고, LK로 잇고, forward–backward·RANSAC(F 검증)으로 거른 특징 트랙이 [BA](slam_math.md)의 재투영 항이 된다. ORB-SLAM은 같은 자리를 디스크립터 매칭으로 채우는 반대파: flow형(연속성 활용, 빠름, drift 누적)과 특징형(재검출 가능, 루프 클로저와 궁합) front-end의 선택은 지금도 유효한 설계 분기점이다. DROID-SLAM은 §9의 RAFT식 flow를 미분가능 BA와 묶어 이 두 파를 학습으로 통합한 사례.

**Visual servoing.** 추적된 이미지 특징 $\mathbf{s}$를 목표 $\mathbf{s}^*$로 보내도록 카메라 속도를 닫는 제어 — 오차 $\mathbf{e} = \mathbf{s} - \mathbf{s}^*$에 대해

$$\dot{\mathbf{e}} = L_s \mathbf{v}_c \quad \Longrightarrow \quad \mathbf{v}_c = -\lambda\, L_s^+ \mathbf{e}$$

($L_s$ = interaction matrix, $\mathbf{v}_c$ = 카메라 트위스트). 트래킹 품질이 곧 제어 루프의 센서 품질인 지점으로, 추적 실패 = 제어 발산이다.

**조작(manipulation).** 파지 중 물체의 6-DoF 포즈 추적(모델 기반 ICP류 ↔ 학습 기반 FoundationPose류), 시연 영상의 TAP 궤적 추출(§10) — [perception](perception.md) 파이프라인이 "한 장의 포즈 추정"에서 "시간축의 포즈 유지"로 확장되는 자리다.

**내 축과의 접속.** 스테레오 매칭(공간 대응)과 optical flow(시간 대응)는 비용·집계·정칙화라는 같은 문법을 쓴다 — §3의 structure tensor 조건은 스테레오의 무텍스처 실패와, §4의 평활 항은 SGM의 정칙화와 정확히 대응한다. 한쪽을 깊이 아는 것이 다른 쪽의 실패 예측 능력이 되는 구조다. 그리고 [3DGS](reviews/gs.md) 쪽에서는 동적 장면 재구성(4DGS류)이 결국 "가우시안의 시간축 대응"이라는 트래킹 문제를 내장하고 있다.

## 13. 계보 한 장 정리

| 연도 | 기법 | 한 줄 |
|---|---|---|
| 1981 | Lucas–Kanade / Horn–Schunck | 밝기 항등식 + 부족한 식을 국소 공유/전역 평활로 |
| 1994 | Shi–Tomasi (KLT) | "추적 가능한 곳"의 조건이 코너 검출이 됨 |
| 2003 | mean-shift 트래킹 | 히스토그램 유사도의 경사 상승 |
| 2010 | MOSSE | 주파수 영역 ridge — 폐형식 온라인 학습, 669fps |
| 2015 | KCF | 순환행렬 대각화 + 커널 — 시프트를 공짜 학습 샘플로 |
| 2016 | SiamFC / SORT | correlation의 재료를 학습 / MOT 최소 구성 |
| 2020 | RAFT | all-pairs 볼륨 + GRU 반복 — 최적화 구조의 학습판 |
| 2022 | OSTrack·MixFormer | 추출과 비교를 한 transformer로 |
| 2023 | TAPIR·CoTracker | 장기 점 궤적 — 가림을 통과하는 집단 추적 |
| 2024 | SAM 2 | 프롬프트 기반 비디오 분할 = 메모리 어텐션 트래킹 |

한 줄 결론: **트래킹의 역사는 "부족한 제약을 어디서 빌려오는가"의 역사다** — 이웃 픽셀(LK), 전역 평활(HS), 운동 모델(칼만), 순환 구조(KCF), 대규모 비디오의 사전학습(Siamese·RAFT), 그리고 이웃 궤적(CoTracker)까지. 빌리는 출처가 국소에서 전역으로, 손설계에서 학습으로 옮겨 왔을 뿐 질문은 1981년 그대로다: *같은 것은, 다음 순간 어디에 있는가.*
