# 10. Extrinsic Calibration · PnP

> 내부 파라미터를 알 때, "지금 카메라가 월드 어디에 어떤 자세로 있는가"를 푸는 문제. 현장 캘리브레이션의 본체다.

## PnP (Perspective-n-Point)

3D 점 $\mathbf{X}_i$(월드)와 그 상 $\mathbf{x}_i$(픽셀) $n$쌍에서 $(R,\mathbf{t})$를 추정한다.

- **P3P**: 최소 3점 + 해 4개 → 검증점 1개로 선택. RANSAC의 최소 샘플로 주로 쓰임.
- **DLT (n≥6)**: $P=K[R|\mathbf{t}]$ 전체를 선형으로 — 빠르지만 정밀도 낮음.
- **EPnP (n≥4)**: 제어점 4개의 선형 조합으로 $O(n)$ — 실무 초기해 표준.
- 마무리는 언제나 **재투영 오차 비선형 최적화**(LM). OpenCV `solvePnP` + `solvePnPRefineLM`.

Outlier가 있으면 `solvePnPRansac` — 대응 자체가 의심스러운 현장에선 필수다.

## 각 방법의 유도

세 방법 모두 같은 투영 방정식에서 출발한다. 월드 점 $\mathbf{X}_i$, 픽셀 $\mathbf{x}_i$(동차), 미지수 $(R,\mathbf{t})$:

$$
\lambda_i\,\tilde{\mathbf{x}}_i = K\,(R\,\mathbf{X}_i + \mathbf{t}),
\qquad \lambda_i > 0 \ (\text{깊이})
$$

차이는 **스케일 $\lambda_i$를 어떻게 소거하고, 회전의 비선형성을 어디서 처리하느냐**다.

### DLT — 전부를 선형으로 밀어붙이기

$P = K[R|\mathbf{t}]$를 통째로 미지수(12개 성분, 스케일 모호라 11 DoF)로 둔다. $\tilde{\mathbf{x}}_i \parallel P\tilde{\mathbf{X}}_i$ (평행 = 같은 방향)이므로 외적으로 $\lambda_i$를 소거한다:

$$
\tilde{\mathbf{x}}_i \times P\tilde{\mathbf{X}}_i = \mathbf{0}
$$

$P$의 행을 $\mathbf{p}_1^\top,\mathbf{p}_2^\top,\mathbf{p}_3^\top$, $\tilde{\mathbf{x}}_i=(u_i,v_i,1)$로 쓰면 외적의 세 성분 중 독립인 둘만 남는다:

$$
\begin{aligned}
u_i\,(\mathbf{p}_3^\top\tilde{\mathbf{X}}_i) - \mathbf{p}_1^\top\tilde{\mathbf{X}}_i &= 0\\
v_i\,(\mathbf{p}_3^\top\tilde{\mathbf{X}}_i) - \mathbf{p}_2^\top\tilde{\mathbf{X}}_i &= 0
\end{aligned}
$$

점 하나당 선형식 2개 → 미지수 11개이므로 **최소 6점**(5.5점). 쌓으면 $A\,\mathrm{vec}(P)=\mathbf{0}$, $A\in\mathbb{R}^{2n\times 12}$ — 해는 $\|\mathrm{vec}(P)\|=1$ 제약 아래 $\|A\,\mathrm{vec}(P)\|$ 최소화, 즉 **$A$의 최소 특이벡터**다(평면 피팅·호모그래피와 같은 틀).

$K$를 알 때는 먼저 $\hat{\mathbf{x}}_i = K^{-1}\tilde{\mathbf{x}}_i$로 정규화하고 $[R|\mathbf{t}]$의 12성분에 대해 같은 시스템을 푼다. 선형 해 $\bar{R}$은 일반적으로 회전이 아니므로 **SO(3) 사영**이 필요하다: $\bar{R}=U\Sigma V^\top$이면

$$
R = U V^\top \ (\det \text{보정}), \qquad s = \frac{3}{\mathrm{tr}(\Sigma)},\quad \mathbf{t} \leftarrow s\,\bar{\mathbf{t}}
$$

정밀도가 낮은 이유가 유도에서 보인다: 최소화하는 양이 재투영 오차가 아니라 **대수적 오차**이고, $R$의 직교 구속을 풀어놓고 사후에 사영하기 때문. 수치 조건을 위해 점 좌표 정규화(Hartley)가 사실상 필수다.

### P3P — 코사인 법칙으로 깊이를 먼저 풀기 (Grunert)

3점이면 미지수를 자세가 아니라 **깊이 3개**로 바꾸는 것이 고전 해법이다. 정규화 시선 벡터(단위) $\mathbf{f}_i = K^{-1}\tilde{\mathbf{x}}_i / \|K^{-1}\tilde{\mathbf{x}}_i\|$를 만들면 카메라 좌표의 점은 $s_i\mathbf{f}_i$ ($s_i$ = 깊이). 우리가 아는 것은 두 가지다:

- 시선 사이각: $\cos\alpha = \mathbf{f}_2\!\cdot\!\mathbf{f}_3,\ \cos\beta = \mathbf{f}_1\!\cdot\!\mathbf{f}_3,\ \cos\gamma = \mathbf{f}_1\!\cdot\!\mathbf{f}_2$
- 월드에서의 점 간 거리: $a = \|\mathbf{X}_2-\mathbf{X}_3\|,\ b = \|\mathbf{X}_1-\mathbf{X}_3\|,\ c = \|\mathbf{X}_1-\mathbf{X}_2\|$

삼각형 (카메라 중심, 점 $i$, 점 $j$)에 **코사인 법칙**을 쓰면:

$$
\begin{aligned}
s_2^2 + s_3^2 - 2 s_2 s_3 \cos\alpha &= a^2\\
s_1^2 + s_3^2 - 2 s_1 s_3 \cos\beta &= b^2\\
s_1^2 + s_2^2 - 2 s_1 s_2 \cos\gamma &= c^2
\end{aligned}
$$

2차식 3개, 미지수 3개. 비율 $u = s_2/s_1,\ v = s_3/s_1$을 도입해 $s_1^2$으로 묶으면:

$$
s_1^2(u^2+v^2-2uv\cos\alpha)=a^2,\quad
s_1^2(1+v^2-2v\cos\beta)=b^2,\quad
s_1^2(1+u^2-2u\cos\gamma)=c^2
$$

식끼리 나눠 $s_1^2$을 소거하면 $(u,v)$에 대한 식 2개가 남고, 한쪽(2차식)에서 $u$를 $v$로 표현해 대입하면 **$v$에 대한 4차 방정식**이 된다 — 해석적으로 풀리고, 실근 최대 4개가 "P3P는 해가 최대 4개"의 정체다. 각 근에서 $s_1 = b/\sqrt{1+v^2-2v\cos\beta}$, $s_2 = u s_1$, $s_3 = v s_1$로 깊이를 복원하면 카메라 좌표 점 $s_i\mathbf{f}_i$를 얻고, 월드 점과의 대응에서 $(R,\mathbf{t})$는 아래 절대 자세 문제로 닫힌다. 4번째 점의 재투영 오차로 해를 고른다.

최근 구현(Kneip, Lambda Twist)은 4차식 경유 없이 회전을 직접 파라미터화해 수치 안정성과 속도를 올린 변형 — 구조는 동일하다.

### EPnP — 미지수를 "제어점 4개의 좌표"로 바꾸기

$n$이 커져도 $O(n)$인 이유는 미지수 재파라미터화에 있다. 월드에서 **제어점** $\mathbf{c}_1..\mathbf{c}_4$(보통 점군 무게중심 + PCA 주축 3개)를 잡고, 모든 점을 그 **무게중심 좌표(barycentric)** 로 쓴다:

$$
\mathbf{X}_i = \sum_{j=1}^{4}\alpha_{ij}\,\mathbf{c}_j^{w},\qquad \sum_j \alpha_{ij}=1
$$

핵심 성질: $\alpha_{ij}$는 **강체 변환에 불변**이다(아핀 결합이 보존되므로). 따라서 카메라 좌표에서도 같은 계수로

$$
\mathbf{X}_i^{c} = \sum_j \alpha_{ij}\,\mathbf{c}_j^{c}
$$

가 성립하고, 미지수는 $(R,\mathbf{t})$ 대신 **제어점의 카메라 좌표 12개** $\{\mathbf{c}_j^c=(x_j,y_j,z_j)\}$가 된다. 투영 방정식

$$
w_i \begin{bmatrix}u_i\\ v_i\\ 1\end{bmatrix} = K \sum_j \alpha_{ij}\,\mathbf{c}_j^c
$$

에서 셋째 행이 $w_i = \sum_j \alpha_{ij} z_j$를 주므로 대입해 소거하면, $K=\mathrm{diag}$꼴($f_u,f_v$, 주점 $u_c,v_c$) 기준으로 점당 선형식 2개:

$$
\begin{aligned}
\sum_j \alpha_{ij}\,\big(f_u\, x_j + (u_c - u_i)\, z_j\big) &= 0\\
\sum_j \alpha_{ij}\,\big(f_v\, y_j + (v_c - v_i)\, z_j\big) &= 0
\end{aligned}
$$

쌓으면 $M\mathbf{x} = \mathbf{0}$, $M\in\mathbb{R}^{2n\times 12}$. 해는 $M$의 **영공간**에 있다:

$$
\mathbf{x} = \sum_{k=1}^{N}\beta_k \mathbf{v}_k,\qquad N\in\{1,\dots,4\}
$$

($\mathbf{v}_k$ = 작은 특이값의 특이벡터; 노이즈·점 배치에 따라 유효 차원이 달라 $N$=1~4 케이스를 모두 시도한다.) 남은 미지수 $\beta_k$는 **강체성**에서 나온다 — 제어점 간 거리는 좌표계와 무관하므로:

$$
\|\mathbf{c}_i^c - \mathbf{c}_j^c\|^2 = \|\mathbf{c}_i^w - \mathbf{c}_j^w\|^2 \quad (6\text{개 쌍})
$$

$\beta$의 2차식 6개가 되고, $N$별로 선형화(곱 $\beta_a\beta_b$를 새 변수로)해 풀고 짧은 Gauss-Newton으로 정제한다. $\mathbf{c}_j^c$가 확정되면 $\mathbf{c}_j^w \to \mathbf{c}_j^c$ 대응 4쌍의 절대 자세 문제로 $(R,\mathbf{t})$가 닫힌해로 나온다. 재투영 오차가 가장 작은 $N$ 케이스를 채택.

### 공통 마무리 ① — 절대 자세 (Procrustes/Kabsch)

P3P·EPnP 모두 마지막은 "3D-3D 대응에서 $(R,\mathbf{t})$"로 끝난다. 평면 피팅·ICP와 같은 수학이다. 중심화한 월드 점 $\mathbf{a}_i$, 카메라 점 $\mathbf{b}_i$에 대해

$$
\min_{R\in SO(3)} \sum_i \|R\mathbf{a}_i - \mathbf{b}_i\|^2
\;\Longleftrightarrow\;
\max_R\ \mathrm{tr}\!\big(R\,H\big),\qquad H = \sum_i \mathbf{a}_i \mathbf{b}_i^\top
$$

(전개하면 $\|\mathbf{a}_i\|,\|\mathbf{b}_i\|$ 항은 상수라 교차항만 남는다.) $H = U\Sigma V^\top$이면 $\mathrm{tr}(RU\Sigma V^\top)=\mathrm{tr}(V^\top R U\,\Sigma)$는 $V^\top RU = I$일 때 최대 — 즉

$$
R = V\,\mathrm{diag}(1,1,\det(VU^\top))\,U^\top,\qquad \mathbf{t} = \bar{\mathbf{b}} - R\bar{\mathbf{a}}
$$

($\det$ 보정은 반사 해 방지.)

### 공통 마무리 ② — 재투영 오차 비선형 정제

위의 모든 닫힌해는 **대수적 비용**의 해라서, 최종 답은 진짜 비용인 재투영 오차로 정제한다:

$$
\min_{R,\mathbf{t}} \sum_i \rho\Big(\big\|\pi\big(K(R\mathbf{X}_i+\mathbf{t})\big) - \mathbf{x}_i\big\|^2\Big)
$$

$R$은 축각 $\boldsymbol{\omega}\in\mathfrak{so}(3)$로 최소 파라미터화하고 GN/LM으로 푼다. 자코비안은 사슬 $\frac{\partial \pi}{\partial \mathbf{X}^c}\,[\,\frac{\partial \mathbf{X}^c}{\partial \boldsymbol{\omega}}\ \big|\ I\,]$ ($\pi$의 미분은 $1/z$ 투영의 2×3 행렬). $\rho$는 Huber 등 robust 커널 — RANSAC이 거른 뒤에도 잔존 outlier를 눌러준다.

### 비교 한 장

| 방법 | 최소 점 | 해 개수 | 복잡도 | 자리 |
|---|---|---|---|---|
| DLT | 6 | 1 (대수해) | SVD 1회 | 빠른 초기화, $K$ 미지도 가능 — 정밀도 최저 |
| P3P | 3 | ≤4 | 4차식 | RANSAC 최소 샘플의 표준 |
| EPnP | 4 | 1~케이스 | $O(n)$ | 일반 초기해 표준 |
| GN/LM 정제 | — | 국소 | 반복 | 항상 마지막에 |

## 무엇이 정확도를 정하나

- **점의 공간 분포**: 한 평면·한 구석에 몰린 점은 깊이 방향을 못 잡는다. 화면 전체·깊이 범위를 덮어야 한다.
- **점 개수보다 기하**: 잘 퍼진 8점이 몰린 80점보다 낫다.
- **관측 거리**: 재투영 1 px의 3D 의미는 거리에 비례해 커진다 — W.D 17 m에서 픽셀 정밀도가 mm를 좌우했던 이유.

## 현장에서의 형태

석사 계측 과제의 외부 캘리브레이션이 정확히 이 문제였다:

- **화이트보드(2D 타깃)**: 점 분포가 좋아 정확 — 그러나 대형 W.D에서 크고 무겁다.
- **1D 타깃(막대)**: 휴대성·가시성 우수 — 이론적 근거와 조건은 [1D 캘리브레이션 리뷰](../reviews/1d-calibration.md)에서 자세히.
- 3차 현장 계측 결과 둘의 정확도가 유사함을 확인(평균 2.46 vs 2.68 mm) — "타깃 선택은 정확도가 아니라 운용의 문제"라는 결론.

## 로봇으로 이어지면

카메라-로봇이 함께 있으면 extrinsic은 hand-eye 문제 $AX=XB$가 된다. 포즈 다양성(회전축 2개 이상)이 관측 가능성을 결정한다는 점은 PnP의 "점 분포" 원칙과 같은 이야기다 — 내 실측: 29 poses, 위치 4.2 mm · 자세 0.10°.
