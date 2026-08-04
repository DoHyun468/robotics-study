# SLAM 수식·유도 (수업 자료)

이 장은 [SLAM 실측 트랙](slam.md)의 S0→S6에서 **실제로 구현·사용한 수식을 단계별로 유도**한다. 각 절은 (1) 무엇을 푸는가 → (2) 수식과 유도 → (3) 기호 정의 → (4) 우리 코드의 어디에 쓰였나 순서다.

## 0. 표기와 좌표 규약

- **회전** $R\in SO(3)$: $R^\top R=I,\ \det R=1$. **병진** $t\in\mathbb{R}^3$.
- **강체변환** $T=\begin{bmatrix}R&t\\\mathbf 0^\top&1\end{bmatrix}\in SE(3)$. 동차좌표 $\tilde X=[X;1]$에 대해 $T\tilde X = [RX+t;1]$.
- **월드-카메라 규약**: $T_{wc}$ = *world-from-camera*(카메라 좌표를 월드로), 역은 $T_{cw}=T_{wc}^{-1}$.
- **카메라 프레임(OpenCV)**: $x$ 오른쪽, $y$ 아래, $z$ 광축 전방. (MuJoCo는 $y$ 위, $z$ 후방이라 뒤에서 변환.)
- **hat 연산자** $[\cdot]_\times:\mathbb{R}^3\to\mathfrak{so}(3)$: 벡터를 외적행렬로.

$$[\boldsymbol\omega]_\times=\begin{bmatrix}0&-\omega_3&\omega_2\\\omega_3&0&-\omega_1\\-\omega_2&\omega_1&0\end{bmatrix},\qquad [\boldsymbol\omega]_\times v=\boldsymbol\omega\times v.$$

---

## 1. 카메라 모델과 역투영 (S0)

### 1.1 핀홀 투영
카메라 좌표계의 3D 점 $X_c=(X,Y,Z)$가 이미지 평면 $(u,v)$로 투영되는 관계. 닮은삼각형에서 광축거리 $Z$에 대해 상은 초점거리 $f$만큼 축소되므로

$$u=f_x\frac{X}{Z}+c_x,\qquad v=f_y\frac{Y}{Z}+c_y.$$

동차형으로 쓰면 $\lambda\begin{bmatrix}u\\v\\1\end{bmatrix}=K\,X_c$, 여기서 $\lambda=Z$이고 **내부 파라미터 행렬**

$$K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.$$

- $f_x,f_y$: 픽셀 단위 초점거리, $(c_x,c_y)$: 주점(principal point, 보통 이미지 중심).

### 1.2 화각(fovy)에서 $f$ 유도
MuJoCo는 초점거리 대신 **세로 화각** $\text{fovy}$를 준다. 이미지 높이 $H$(픽셀), 센서 절반 높이 $H/2$가 광축과 이루는 각이 $\text{fovy}/2$이므로

$$\tan\!\Big(\frac{\text{fovy}}{2}\Big)=\frac{H/2}{f}\ \Longrightarrow\ f=\frac{H/2}{\tan(\text{fovy}/2)}.$$

우리 코드는 정사각 픽셀 가정으로 $f_x=f_y=f$, $c_x=W/2,\ c_y=H/2$.

### 1.3 역투영 (2D+depth → 3D)
투영을 뒤집는다. depth $z=Z$가 주어지면 $X=\dfrac{(u-c_x)z}{f_x}$, $Y=\dfrac{(v-c_y)z}{f_y}$, 즉

$$X_c=z\,K^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix}=\Big(\tfrac{(u-c_x)z}{f_x},\ \tfrac{(v-c_y)z}{f_y},\ z\Big),\qquad X_w=R_{wc}X_c+t_{wc}.$$

### 1.4 MuJoCo→OpenCV 축변환
MuJoCo 카메라축을 OpenCV로 돌리는 고정행렬

$$M_{2cv}=\mathrm{diag}(1,-1,-1),\qquad R_{wc}=R_{\text{mj}}\,M_{2cv}.$$

**코드:** `slam_seq.py`의 `gt_KTwc()`(K와 $T_{wc}$), `slam_vo.py`의 `backproject()`.

---

## 2. Point-to-plane ICP — scan-to-scan 오도메트리 (S1)

### 2.1 문제
두 프레임의 포인트클라우드(소스 $\{p_i\}$, 타깃 $\{q_i\}$, 타깃 법선 $\{n_i\}$)를 정합하는 강체변환 $(R,t)$를 찾는다. **point-to-plane**은 점을 점이 아니라 **타깃의 접평면**에 맞추므로 평면 위 미끄러짐을 허용해 수렴이 빠르다:

$$\min_{R,t}\ \sum_i\Big(\big(Rp_i+t-q_i\big)\cdot n_i\Big)^2.$$

### 2.2 소각 선형화
한 반복에서 증분 회전이 작다고 보고 $R\approx I+[\boldsymbol\omega]_\times$ (1차 근사). 그러면 $Rp_i\approx p_i+\boldsymbol\omega\times p_i=p_i-p_i\times\boldsymbol\omega=p_i-[p_i]_\times\boldsymbol\omega$. 잔차의 법선방향 성분은

$$
\big(Rp_i+t-q_i\big)\cdot n_i
\approx \underbrace{(p_i-q_i)\cdot n_i}_{r_i}
+\big(\boldsymbol\omega\times p_i\big)\cdot n_i + t\cdot n_i.
$$

스칼라 삼중곱 항등식 $(\boldsymbol\omega\times p_i)\cdot n_i=\boldsymbol\omega\cdot(p_i\times n_i)$을 쓰면, 미지수 $x=[\boldsymbol\omega;\,t]\in\mathbb{R}^6$에 대해 **선형**이 된다:

$$
\big(Rp_i+t-q_i\big)\cdot n_i\approx r_i + \underbrace{[\,(p_i\times n_i)^\top\ \ n_i^\top\,]}_{G_i\in\mathbb{R}^{1\times6}}\,x.
$$

### 2.3 정규방정식
$\sum_i (r_i+G_i x)^2$를 최소화 → $x$로 미분해 0:

$$\Big(\sum_i G_i^\top G_i\Big)x=-\sum_i G_i^\top r_i,\qquad\text{즉}\quad A x=b,\ \ A=\sum_iG_i^\top G_i,\ b=-\sum_iG_i^\top r_i.$$

$6\times6$ 선형계 한 번 풀어 $x=[\boldsymbol\omega;t]$를 얻고, $\Delta T=\exp([\,\boldsymbol\omega;t\,])$(§6의 SE(3) 지수)로 $T\leftarrow \Delta T\,T$ 갱신. KD-tree로 최근접 대응 재계산하며 수렴까지 반복. 법선 $n_i$는 open3d로 이웃 PCA 추정.

**궤적 적분:** ICP가 소스(프레임 $i{+}1$)를 타깃(프레임 $i$) 좌표로 보내는 $M$을 주면, 카메라 pose는 $T_{wc}^{(i+1)}=T_{wc}^{(i)}\,M$로 누적.

**코드:** `slam_vo.py`의 `icp_point_to_plane()`, `estimate_normals()`, `vo_icp()`.

---

## 3. ORB + PnP — 특징 기반 오도메트리 (S1)

### 3.1 특징 매칭
프레임마다 **ORB**(FAST 코너 + BRIEF 이진 기술자) 키포인트를 뽑고, 해밍 거리 + cross-check로 두 프레임을 매칭.

### 3.2 PnP (Perspective-n-Point)
프레임 $i$의 매칭 키포인트를 depth로 3D화한 $\{X_i\}$(카메라 $i$ 좌표)와, 프레임 $i{+}1$에서의 2D 관측 $\{u_i\}$가 주어질 때, 재투영오차를 최소화하는 상대 pose $(R,t)$(카메라 $i$→$i{+}1$):

$$\min_{R,t}\sum_i\big\|\,\pi\!\big(RX_i+t\big)-u_i\,\big\|^2,\qquad \pi(X)=\Big(f_x\tfrac{X}{Z}+c_x,\ f_y\tfrac{Y}{Z}+c_y\Big).$$

- $\pi(\cdot)$: 핀홀 투영(§1.1). Levenberg–Marquardt로 비선형 최소자승을 풀고, **RANSAC**으로 잘못된 매칭(outlier)을 걸러 inlier에서만 추정.
- 결과 $T_{i{+}1\leftarrow i}$의 역이 §2의 $M$(카메라 $i{+}1$→$i$ 좌표)이 되어 같은 방식으로 적분.

**코드:** `slam_vo.py`의 `vo_pnp()`(`cv2.solvePnPRansac`), 루프클로저용 `pnp_relative()`(`slam_posegraph.py`).

---

## 4. 궤적 평가 — Umeyama 정렬, ATE, RPE (S1)

### 4.1 Umeyama 정렬 (왜 필요한가)
추정 궤적은 시작 좌표계·(mono라면)스케일이 GT와 다를 수 있다. 두 점집합 $\{e_i\}$(추정)·$\{g_i\}$(GT)를 맞추는 최적 닮음변환 $(s,R,t)$:

$$\min_{s,R,t}\sum_i\big\|\,sRe_i+t-g_i\,\big\|^2.$$

평균 제거 $\bar e,\bar g$ 후 공분산 $\Sigma=\tfrac1N\sum(g_i-\bar g)(e_i-\bar e)^\top=UDV^\top$(SVD). 해는

$$R=U\,S\,V^\top,\quad S=\mathrm{diag}(1,\dots,1,\det(UV^\top)),\qquad t=\bar g-sR\bar e.$$

$S$의 마지막 원소가 반사(개선된 해가 회전이 아닌 반사가 되는 것)를 막는다. RGB-D는 metric이라 $s=1$로 고정(SE(3) 정렬).

### 4.2 지표
- **ATE(Absolute Trajectory Error):** 정렬 후 전역 위치 RMSE

$$\text{ATE}=\sqrt{\tfrac1N\textstyle\sum_i\|\,sRe_i+t-g_i\|^2}.$$

- **RPE(Relative Pose Error, 프레임 간격 $\Delta$):** 국소 상대 모션의 오차(drift율)

$$E_i=\big(T^{g}_i{}^{-1}T^{g}_{i+\Delta}\big)^{-1}\big(T^{e}_i{}^{-1}T^{e}_{i+\Delta}\big),\quad
\text{병진}=\|E_i^{t}\|,\ \ \text{회전}=\arccos\!\frac{\mathrm{tr}(E_i^R)-1}{2}.$$

회전각 공식은 $\mathrm{tr}(R)=1+2\cos\theta$에서 나온다.

**코드:** `slam_vo.py`의 `umeyama()`, `ate()`, `rpe()`, `rot_angle()`.

---

## 5. TSDF 융합 — 맵핑 (S2)

### 5.1 절단 부호거리장(TSDF)
각 voxel은 가장 가까운 표면까지의 **부호거리** $d$(표면 앞 음수/뒤 양수)를 절단해 저장:

$$\Psi(d)=\max\!\big(-1,\min(1,\,d/\tau)\big),\qquad \tau=\text{truncation}.$$

### 5.2 다중 프레임 가중 융합
프레임마다 관측 $\Psi$를 가중 이동평균으로 누적(각 관측 신뢰도 $w$):

$$D\leftarrow\frac{W D+w\,\Psi}{W+w},\qquad W\leftarrow W+w.$$

표면은 $D=0$ 영교차면(marching cubes로 삼각망 추출). 카메라 extrinsic은 $T_{cw}=T_{wc}^{-1}$.

### 5.3 맵 품질 — 대칭 Chamfer
$$\text{Chamfer}=\tfrac12\Big(\underbrace{\overline{\min_j\|a_i-b_j\|}}_{\text{정확도, est}\to\text{gt}}+\underbrace{\overline{\min_i\|b_j-a_i\|}}_{\text{완전성, gt}\to\text{est}}\Big).$$

바닥 평면은 yaw 회전에 거의 불변이라 씬 구조를 가리므로 **평가에서 크롭**(정직 메모).

**코드:** `slam_map.py`의 `fuse_tsdf()`(open3d `ScalableTSDFVolume`), `map_error_mm()`.

---

## 6. SE(3) 리군과 Pose-graph 최적화 (S3)

### 6.1 SO(3) 지수사상 (Rodrigues)
축각 $\boldsymbol\phi=\theta\,\hat k$($\theta=\|\boldsymbol\phi\|$)에 대한 회전:

$$R=\exp([\boldsymbol\phi]_\times)=I+\frac{\sin\theta}{\theta}[\boldsymbol\phi]_\times+\frac{1-\cos\theta}{\theta^2}[\boldsymbol\phi]_\times^2.$$

$[\hat k]_\times^3=-[\hat k]_\times$라는 성질로 지수급수를 sin·cos로 접은 결과다. 역(log): $\theta=\arccos\!\frac{\mathrm{tr}(R)-1}{2}$, $[\boldsymbol\phi]_\times=\frac{\theta}{2\sin\theta}(R-R^\top)$.

### 6.2 SE(3) 지수사상
$\boldsymbol\xi=[\boldsymbol\rho;\boldsymbol\phi]\in\mathbb{R}^6$($\boldsymbol\rho$=병진성분, $\boldsymbol\phi$=회전성분):

$$\exp(\boldsymbol\xi)=\begin{bmatrix}R&V\boldsymbol\rho\\\mathbf0^\top&1\end{bmatrix},\quad
V=I+\frac{1-\cos\theta}{\theta^2}[\boldsymbol\phi]_\times+\frac{\theta-\sin\theta}{\theta^3}[\boldsymbol\phi]_\times^2.$$

$V$는 회전이 병진에 미치는 결합을 담는 좌 야코비안. 역(log)은 $\boldsymbol\phi=\log R$, $\boldsymbol\rho=V^{-1}t$.

### 6.3 Adjoint
좌표 프레임 사이 twist를 옮기는 $6\times6$ 행렬:

$$\mathrm{Ad}_T=\begin{bmatrix}R&[t]_\times R\\\mathbf0&R\end{bmatrix},\qquad \exp\!\big(\mathrm{Ad}_T\,\boldsymbol\xi\big)=T\exp(\boldsymbol\xi)T^{-1}.$$

### 6.4 Pose-graph: 노드·엣지·잔차
- **노드**: 카메라 pose $\{T_i\}$. **엣지(측정)**: 상대 pose $Z_{ij}$ — 연속 프레임은 오도메트리(S1), 재방문 프레임쌍은 **loop closure**(ORB+PnP).
- **BetweenFactor 잔차**: 측정과 현재 추정의 불일치를 리대수에서

$$e_{ij}=\log\!\big(Z_{ij}^{-1}\,T_i^{-1}T_j\big)\in\mathbb{R}^6.$$

최적점에서 $T_i^{-1}T_j=Z_{ij}$이면 $e_{ij}=0$.

### 6.5 Jacobian 유도 (우섭동)
$T_i\!\leftarrow\!T_i\exp(\boldsymbol\delta_i)$, $T_j\!\leftarrow\!T_j\exp(\boldsymbol\delta_j)$로 섭동하고 $A=Z_{ij}^{-1}T_i^{-1}T_j$라 하자. 항등식 $\exp(\boldsymbol\delta)X=X\exp(\mathrm{Ad}_{X^{-1}}\boldsymbol\delta)$을 이용하면

$$e_{ij}(\boldsymbol\delta)=\log\!\Big(A\,\exp\big(-\mathrm{Ad}_{T_j^{-1}T_i}\boldsymbol\delta_i+\boldsymbol\delta_j\big)\Big)\approx e_{ij}+J_r^{-1}(e_{ij})\big(-\mathrm{Ad}_{T_j^{-1}T_i}\boldsymbol\delta_i+\boldsymbol\delta_j\big).$$

잔차가 작을 때 우야코비안 $J_r^{-1}\!\approx\!I$로 근사하면

$$\frac{\partial e_{ij}}{\partial\boldsymbol\delta_i}=-\mathrm{Ad}_{T_j^{-1}T_i},\qquad \frac{\partial e_{ij}}{\partial\boldsymbol\delta_j}=I.$$

### 6.6 Gauss-Newton
모든 엣지의 정보행렬 가중 $W_e$로

$$H=\sum_e J_e^\top W_e J_e,\quad g=\sum_e J_e^\top W_e e_e,\qquad H\,\boldsymbol\delta=-g,$$

풀어 각 노드를 **retraction** $T_k\leftarrow T_k\exp(\boldsymbol\delta_k)$로 갱신(앵커 노드 0은 고정). loop closure 엣지가 누적 drift를 전역적으로 접어 ATE를 낮춘다. 수치 안정용 LM 감쇠 $H+\lambda I$.

**코드:** `slam_posegraph.py`의 `se3_exp/se3_log/adjoint`, `optimize()`, `detect_loops()`.

---

## 7. NeRF — 신경 볼륨 렌더링 (S4)

### 7.1 볼륨 렌더링 적분
광선 $r(t)=o+t\,d$를 따라 방사휘도를 누적. 투과율(transmittance) $T(t)=\exp\!\big(-\int_{t_n}^{t}\sigma(r(s))\,ds\big)$일 때 색은

$$C=\int_{t_n}^{t_f} T(t)\,\sigma(r(t))\,c(r(t))\,dt.$$

### 7.2 이산화(알파 합성)
샘플 $t_1<\dots<t_N$, 간격 $\delta_i=t_{i+1}-t_i$에서 $\alpha_i=1-e^{-\sigma_i\delta_i}$(구간 흡수 확률), 누적 투과율 $T_i=\prod_{j<i}(1-\alpha_j)$. 그러면

$$\hat C=\sum_i T_i\,\alpha_i\,c_i,\qquad \hat D=\sum_i T_i\,\alpha_i\,t_i\ (\text{depth}).$$

MLP가 위치 $x$에서 밀도 $\sigma$와 색 $c$를 출력. 고주파 표현을 위해 **positional encoding**

$$\gamma(x)=\big(\,x,\ \sin(2^0\pi x),\cos(2^0\pi x),\ \dots,\ \sin(2^{L-1}\pi x),\cos(2^{L-1}\pi x)\,\big).$$

손실 = photometric $\|\hat C-C\|^2$ + depth $|\hat D-D|$(RGB-D 감독으로 수렴 가속).

**코드:** `slam_nerf.py`의 `render_rays()`, `PosEnc`, `NeRF`.

---

## 8. 3D Gaussian Splatting (S4b)

### 8.1 장면 표현
장면 = 3D 가우시안들의 합. 가우시안 $i$: 평균 $\mu_i$, 공분산 $\Sigma_i$, 색 $c_i$, 불투명도 $o_i$.

$$G_i(x)=\exp\!\Big(-\tfrac12(x-\mu_i)^\top\Sigma_i^{-1}(x-\mu_i)\Big).$$

공분산은 항상 양의 정부호이도록 회전 $R$·스케일 $S$로 분해해 파라미터화:

$$\Sigma=R\,S\,S^\top R^\top\quad(S=\mathrm{diag}(s_x,s_y,s_z)).$$

### 8.2 2D 투영 (EWA)
카메라로 투영 시 공분산은 투영 야코비안 $J$와 뷰 회전 $W$로 근사 변환(EWA splatting):

$$\Sigma'=J\,W\,\Sigma\,W^\top J^\top.$$

### 8.3 알파 합성
픽셀에서 깊이순으로 정렬된 가우시안을 합성:

$$C=\sum_i c_i\,\alpha_i\prod_{j<i}(1-\alpha_j),\qquad \alpha_i=o_i\,G_i^{2D}(\text{pixel}).$$

미분가능 rasterizer로 photometric 손실을 $\{\mu,\Sigma(R,S),c,o\}$에 역전파. 우리는 등방 가우시안 $\Sigma=s^2I$로 단순화(방향 무관)하고, gaussian을 RGB-D back-projection으로 초기화.

**코드:** `slam_3dgs.py`(gsplat `rasterization`), `init_gaussians()`, `train_3dgs()`.

---

## 9. SplaTAM식 tracking (S6)

### 9.1 photometric pose 최적화
맵(가우시안)을 고정하고 카메라 pose만 렌더링 오차로 추정:

$$T^\star=\arg\min_{T}\ \big\|\,\mathcal R(\text{map};\,T)-I_{\text{obs}}\,\big\|_1,\qquad \mathcal R=\text{미분가능 렌더}.$$

### 9.2 등방 가우시안 단순화 (왜 means 변환과 동치인가)
카메라를 $T$로 움직여 렌더하는 것은, 카메라를 고정($V_{\text{init}}$)한 채 **월드(가우시안)를 반대로 움직여** 렌더하는 것과 같다. 가우시안 평균을 강체변환 $\mu_i'=\hat R\mu_i+\hat t$하면 유효 뷰행렬은 $V_{\text{init}}\begin{bmatrix}\hat R&\hat t\\\mathbf0&1\end{bmatrix}$. 등방이라 방향(공분산)은 변환에 불변이므로 means만 옮기면 충분하다. $\hat R=\exp([\boldsymbol\omega]_\times)$의 $(\boldsymbol\omega,\hat t)$를 Adam으로 최적화. 복원 pose $T_{wc}=\big(V_{\text{init}}[\hat R\,|\,\hat t]\big)^{-1}$.

**코드:** `slam_track.py`의 `axis_angle_to_matrix()`, `render_frozen()`, `track_frame()`.

---

## 10. 공통 지표

- **PSNR**(렌더 품질): $\text{PSNR}=10\log_{10}\!\dfrac{\text{peak}^2}{\text{MSE}}$ (peak=1, 정규화 이미지).

---

*각 단계의 실측 수치·그림은 [SLAM 트랙](slam.md), 코드는 `robotics-lab/src/`, 단계별 노트는 `robotics-lab/notes/slam_*.md`.*
