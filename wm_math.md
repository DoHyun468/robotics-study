# World Models 수식·유도 (수업 자료)

이 장은 [World Models hands-on 트랙](wm.md)의 W0→W5에서 **실제로 구현·사용한 수식을 단계별로 유도**한다. 각 절은 (1) 무엇을 푸는가 → (2) 수식과 유도 → (3) 기호 정의 → (4) 우리 코드의 어디에 쓰였나 순서다. 표기는 [SLAM 수식 챕터](slam_math.md)의 $SE(3)$ 규약을 그대로 잇는다.

## 0. 공통 셋업 — GT action과 (context→future) 문제

world model은 관측열 $o_{1:t}$ 과 행동열 $a_{1:t-1}$ 에서 미래 $o_{t+1:t+h}$(또는 그 표현)을 예측한다. 우리는 SLAM S0 시퀀스를 재사용하되, **행동을 GT pose에서 정확히 뽑는다**.

### 0.1 카메라 트위스트를 action으로
인접 프레임의 상대 카메라 운동을 *현재 카메라 프레임*에서 본 것이 action이다.

$$T_{t\to t+1}=T_{wc}(t)^{-1}\,T_{wc}(t+1)\in SE(3),\qquad a_t=\log\!\big(T_{t\to t+1}\big)^\vee\in\mathbb{R}^6 .$$

$\log:SE(3)\to\mathfrak{se}(3)$ 의 닫힌형은 §4에서 유도한다. 등속 궤도라 $\lVert a_t\rVert$ 는 거의 상수(측정값 $\approx0.075$).

- **기호**: $T_{wc}(t)$ world-from-camera pose, $a_t=[v;\,\omega]$ 병진·회전 트위스트.
- **코드**: `wm_data.py · se3_log()`, `load_sequence()`가 프레임쌍마다 `actions[t]` 생성.

### 0.2 평가 지표
- 표현 예측: 정규화 latent-MSE. 픽셀 예측: PSNR $=10\log_{10}(1/\mathrm{MSE})$, SSIM(§5).
- **horizon 곡선**: $h$ 스텝 앞 예측의 지표를 $h$의 함수로 — "몇 스텝까지 버티나"를 본다.
- **코드**: `wm_data.py · psnr(), ssim(), horizon_curve()`.

---

## 1. W1 — JEPA (예측·표현형)

### 1.1 무엇을 푸는가
픽셀을 복원하지 않고 **미래 프레임의 표현을 예측**한다. 관측 $x$를 잠재 $z=f_\theta(x)$로 인코딩하고, action을 받아 잠재를 굴리는 예측기 $p_\phi$ 를 학습한다. 목표(target) 표현은 **EMA 인코더** $f_\xi$ 가 준다.

### 1.2 잠재 동역학 (action-conditioned residual)
예측기는 잔차 형태로 한 스텝을 굴린다.

$$\hat z_{t+1}=\hat z_t+g_\phi\!\big([\hat z_t,\,a_t]\big),\qquad \hat z_{t+1:t+h}=\underbrace{p_\phi(z_k,\,a_{k:k+h-1})}_{h\text{-step rollout}} .$$

시작 상태 $z_k=f_\theta(x_k)$ 는 context 마지막 프레임의 online 임베딩.

### 1.3 목적함수 — BYOL식 정규화 예측 + collapse 방지
target은 stop-grad된 EMA 인코더의 미래 임베딩 $z^{\text{tgt}}_{t}=f_\xi(x_{t})$. 각 벡터를 $\ell_2$ 정규화($\bar z=z/\lVert z\rVert$)한 뒤 MSE:

$$\mathcal L_{\text{pred}}=\frac1h\sum_{i=1}^{h}\big\lVert \overline{\hat z_{k+i}}-\overline{z^{\text{tgt}}_{k+i}}\big\rVert_2^2 .$$

정규화 MSE는 코사인 유사도와 등가다: $\lVert\bar a-\bar b\rVert^2=2-2\,\bar a^\top\bar b=2(1-\cos\angle(a,b))$.

**왜 붕괴(collapse)하지 않나.** 만약 online과 target이 같은 네트워크였다면 $f\equiv\text{const}$ 가 손실 0의 자명해다. EMA(§1.4)로 target을 online의 **느린 이동평균**으로 두면, 예측기가 target을 따라잡는 동안 target이 천천히 움직여 자명해로의 붕괴를 깬다(BYOL의 핵심). 추가 안전장치로 **VICReg 분산 힌지**를 online 임베딩에 건다.

$$\mathcal L_{\text{var}}=\frac1D\sum_{d=1}^{D}\max\!\big(0,\ 1-\sqrt{\operatorname{Var}(z_{\cdot,d})+\epsilon}\big),\qquad \mathcal L=\mathcal L_{\text{pred}}+\lambda\,\mathcal L_{\text{var}} .$$

각 차원의 배치 표준편차가 1 미만이면 벌점 → 표현이 한 점으로 뭉치지 못하게 밀어낸다. 학습이 끝나면 $\mathcal L_{\text{var}}\to0$(측정값), 즉 붕괴 없음.

### 1.4 EMA 업데이트
$$\xi \leftarrow m\,\xi + (1-m)\,\theta,\qquad m=0.996 .$$
$\theta$ 는 gradient로, $\xi$ 는 위 식으로만 갱신(gradient 흐르지 않음).

- **기호**: $f_\theta$ online, $f_\xi$ target 인코더, $g_\phi$ 잔차 동역학, $D$ 임베딩 차원, $m$ 모멘텀, $\lambda$ 분산 가중치.
- **코드**: `wm_jepa.py · Encoder, Predictor.rollout(), vicreg_var(), ema_update(), train()`.

### 1.5 선형 probe와 effective rank
얼린 임베딩 $Z\in\mathbb R^{T\times D}$ 가 기하량 $y$(카메라 위치, depth)를 선형으로 담는지 본다. §5.3의 held-out ridge $R^2$ 로 측정. 표현의 내재 차원은 **effective rank**(특이값 스펙트럼의 엔트로피 지수)로:

$$\sigma=\text{svd}(Z-\bar Z),\quad p_i=\frac{\sigma_i}{\sum_j\sigma_j},\quad \operatorname{erank}(Z)=\exp\!\Big(-\sum_i p_i\log p_i\Big).$$

우리 궤도는 1-자유도라 $\operatorname{erank}\approx6\ll D=256$ — 붕괴가 아니라 데이터의 진짜 저차원성.

---

## 2. W2 — RSSM (시퀀스·토큰형, PlaNet/Dreamer)

### 2.1 생성 모델
잠재 상태를 **결정론 $h_t$ + 확률 $s_t$** 로 나눈 순환 상태공간모델(RSSM):

$$h_t=\text{GRU}\big(h_{t-1},\,[s_{t-1},a_{t-1}]\big),\quad
\begin{cases}\text{prior: } p_\theta(s_t\mid h_t)=\mathcal N(\mu_p(h_t),\sigma_p(h_t))\\[2pt]
\text{decoder: } p_\theta(x_t\mid h_t,s_t)=\mathcal N(D(h_t,s_t),\,I)\end{cases}$$

추론용 **posterior**는 관측 임베딩 $e_t=\text{enc}(x_t)$ 를 함께 본다: $q_\phi(s_t\mid h_t,e_t)=\mathcal N(\mu_q,\sigma_q)$.

### 2.2 ELBO 유도
로그가능도의 변분하한(자유에너지)을 시퀀스에 대해 쓴다. 한 스텝의 증거하한은

$$\log p_\theta(x_t)\ \ge\ \underbrace{\mathbb E_{q_\phi(s_t)}\big[\log p_\theta(x_t\mid h_t,s_t)\big]}_{\text{복원(reconstruction)}}\ -\ \underbrace{\mathrm{KL}\!\big(q_\phi(s_t\mid h_t,e_t)\,\Vert\,p_\theta(s_t\mid h_t)\big)}_{\text{동역학 일치(dynamics KL)}} .$$

유도: $\log p(x)=\log\int p(x\mid s)p(s)\,ds=\log\mathbb E_{q}\!\big[\tfrac{p(x\mid s)p(s)}{q(s)}\big]\ge\mathbb E_q[\log p(x\mid s)]-\mathrm{KL}(q\Vert p)$ (Jensen). 가우시안 복원은 $-\log p(x\mid s)=\tfrac12\lVert x-D(h,s)\rVert^2+\text{const}$ → **MSE 복원항**.

전체 손실(최소화):

$$\mathcal L_{\text{RSSM}}=\frac1L\sum_{t=1}^{L}\Big[\tfrac12\lVert x_t-D(h_t,s_t)\rVert^2\ +\ \beta\,\max\big(\mathrm{KL}_t,\ \text{free\_nats}\big)\Big].$$

### 2.3 두 가우시안의 KL과 free-bits
대각 가우시안 $q=\mathcal N(\mu_q,\sigma_q^2),\ p=\mathcal N(\mu_p,\sigma_p^2)$ 의 KL은 차원합:

$$\mathrm{KL}(q\Vert p)=\sum_{d}\Big[\log\frac{\sigma_{p,d}}{\sigma_{q,d}}+\frac{\sigma_{q,d}^2+(\mu_{q,d}-\mu_{p,d})^2}{2\sigma_{p,d}^2}-\frac12\Big].$$

**free-bits**: $\max(\mathrm{KL},\ \text{free\_nats})$ 로 KL이 바닥(1 nat) 밑으로 내려가면 벌점을 끊는다 → posterior가 prior로 붕괴(정보 0)하는 것을 막고 잠재가 실제 정보를 담게 한다. 측정에서 KL이 ~2.2 nat에 정착 = 붕괴 없음.

### 2.4 reparameterization과 상상(imagination)
학습 시 $s_t=\mu_q+\sigma_q\odot\varepsilon,\ \varepsilon\sim\mathcal N(0,I)$ 로 미분 가능하게 샘플. **평가(상상)**: 앞 $k$프레임은 posterior로 상태를 세우고(burn-in), 이후 $h$스텝은 **관측 없이 prior만** action으로 굴려 디코드:

$$s_t\leftarrow\mu_p(h_t)\ \ (t>k),\qquad \hat x_t=D(h_t,s_t).$$

- **기호**: $h_t$ 결정론 은닉, $s_t$ 확률 상태, $e_t$ 관측 임베딩, $\beta$ KL 가중, $L=k+h$.
- **코드**: `wm_rssm.py · RSSM.img_step()/obs_step(), kl(), train(), imagine()`.

---

## 3. W3 — 조건부 Diffusion (생성·영상형, DIAMOND식)

### 3.1 forward 과정 (노이즈 주입)
목표 프레임 $x_0$ 에 가우시안 노이즈를 점진적으로 더한다. $\beta_t$ 스케줄, $\alpha_t=1-\beta_t,\ \bar\alpha_t=\prod_{s\le t}\alpha_s$:

$$q(x_t\mid x_{t-1})=\mathcal N\!\big(\sqrt{\alpha_t}\,x_{t-1},\,\beta_t I\big)\ \Longrightarrow\ q(x_t\mid x_0)=\mathcal N\!\big(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\big).$$

닫힌형 유도(가우시안 합성): 재귀적으로 대입하면 평균은 $\sqrt{\bar\alpha_t}x_0$, 분산은 $1-\bar\alpha_t$ 로 접힌다. 따라서 한 방에 샘플 가능:

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

### 3.2 목적함수 (ε-prediction)
reverse $p_\theta(x_{t-1}\mid x_t)$ 를 학습하는 변분하한은, 재파라미터화하면 **노이즈 회귀**로 단순화된다(Ho et al. simplified objective). 조건 $c=[x_{\text{prev}},a]$:

$$\mathcal L_{\text{diff}}=\mathbb E_{x_0,\,t,\,\epsilon}\big\lVert \epsilon-\epsilon_\theta(x_t,\,t,\,c)\big\rVert_2^2 .$$

즉 U-Net $\epsilon_\theta$ 가 "지금 낀 노이즈"를 맞히게 학습. 조건 $c$ 는 직전 프레임(입력에 concat) + action(FiLM/embedding)으로 주입 → **action-conditioned** 생성.

### 3.3 DDIM 결정론 샘플링
학습된 $\epsilon_\theta$ 로 역과정을 돈다. 먼저 $x_t$ 에서 $x_0$ 추정:

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\;\epsilon_\theta(x_t,t,c)}{\sqrt{\bar\alpha_t}} .$$

DDIM은 non-Markovian 결정론 업데이트로 다음(더 낮은 노이즈) 레벨 $t'$ 로 간다(분산항 0):

$$x_{t'}=\sqrt{\bar\alpha_{t'}}\,\hat x_0+\sqrt{1-\bar\alpha_{t'}}\;\epsilon_\theta(x_t,t,c).$$

적은 스텝(우리 30)으로 결정론 생성이 가능해 rollout에 유리.

### 3.4 rollout: teacher-forced vs autoregressive
- **teacher-forced**: 매 스텝 조건 $x_{\text{prev}}$ 를 **GT** 로 → 단일스텝 품질만 측정(오차 누적 없음).
- **autoregressive**: 조건을 **자기 생성물** 로 되먹임 → 오차가 다음 조건을 오염시켜 **drift** 누적. 측정에서 PSNR 30.6→22.1 dB.

- **기호**: $\bar\alpha_t$ 누적 신호비, $\epsilon_\theta$ 노이즈 예측 U-Net, $c$ 조건(직전 프레임+action).
- **코드**: `wm_diffusion.py · Diffusion.q_sample()/ddim(), UNet, train(), rollout()`.

---

## 4. se(3) 지수·로그 (action 추출)

action $a_t=\log(T_{t\to t+1})^\vee$ 계산에 쓴 닫힌형. 회전은 Rodrigues 로그, 병진은 좌-야코비안 역.

### 4.1 SO(3) 로그
$R\in SO(3)$ 의 회전각 $\theta=\arccos\!\big(\tfrac{\operatorname{tr}R-1}{2}\big)$, 축 $\hat\omega$ 는

$$[\hat\omega]_\times=\frac{\theta}{2\sin\theta}\,(R-R^\top),\qquad \omega=\theta\,\hat\omega .$$

$\theta\to0$ 이면 1차 근사 $\omega=\tfrac12\,\text{vee}(R-R^\top)$.

### 4.2 SE(3) 로그 — 병진 성분
$T=\begin{bmatrix}R&t\\0&1\end{bmatrix}$ 에 대해 $\exp([v;\omega]^\wedge)=T$ 를 풀면 $t=V(\omega)\,v$, 즉 $v=V^{-1}t$. 좌-야코비안 역의 닫힌형($K=[\hat\omega]_\times$):

$$V^{-1}=I-\tfrac12\,[\omega]_\times+\Big(\tfrac1{\theta^2}-\tfrac{1+\cos\theta}{2\theta\sin\theta}\Big)[\omega]_\times^2 .$$

(구현은 수치적으로 안정한 등가형 $\;I-\tfrac12\theta K+(1-\tfrac{\theta}{2}\cot\tfrac{\theta}{2})K^2,\ K=[\hat\omega]_\times$ 를 사용.)

- **코드**: `wm_data.py · so3_log(), se3_log()`.

---

## 5. 메트릭 유도

### 5.1 PSNR
$[0,1]$ 이미지에서 최대값 1이므로 $\text{PSNR}=10\log_{10}\!\big(\tfrac{1}{\text{MSE}}\big)$ dB. MSE가 작을수록 dB↑.

### 5.2 SSIM
국소 창(가우시안 가중)에서 밝기·대비·구조를 결합:

$$\text{SSIM}(x,y)=\frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)},\quad c_1=0.01^2,\ c_2=0.03^2 .$$

$\mu,\sigma$ 는 창 내 평균·(공)분산. 1에 가까울수록 구조적으로 동일.

### 5.3 held-out ridge probe와 $R^2$
얼린 표현 $X$ 로 $y$ 를 선형 예측: $\hat W=\arg\min\lVert Xw-y\rVert^2+\lambda\lVert w\rVert^2$, 정규방정식

$$\hat W=(X^\top X+\lambda I)^{-1}X^\top y .$$

과적합(특징수 > 표본수) 차단을 위해 **PCA로 16차원**으로 줄인 뒤 **k-fold hold-out** 에서만 평가:

$$R^2=1-\frac{\sum_{\text{test}}(y-\hat y)^2}{\sum_{\text{test}}(y-\bar y_{\text{train}})^2}.$$

안 본 프레임에서 $R^2$ 가 높으면 "표현이 그 양을 선형으로 담는다"는 주장이 정직하게 성립.

- **코드**: `wm_jepa.py · _pca(), ridge_probe_cv(), collapse_stats()`; probe는 W1·W4 공용.

---

## 6. 한 줄 정리

세 계열은 **"무엇을 예측하느냐"** 로 갈린다 — 표현(JEPA, $\mathcal L_{\text{pred}}$), 잠재상태의 동역학(RSSM, ELBO=복원+KL), 픽셀(diffusion, ε-회귀). 공통 신호는 GT 카메라 트위스트 action이고, 공통 잣대는 GT로 매긴 horizon 곡선·probe $R^2$ 다. 잠재공간 rollout이 픽셀공간보다 drift에 강하다는 것(W2 vs W3)과, 예측을 위한 표현이 기하를 가장 깨끗이 담는다는 것(W1 probe)이 수식이 예고하고 측정이 확인한 핵심이다.
