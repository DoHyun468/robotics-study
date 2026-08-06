# Camera Calibration with 1D Objects (Zhang, 2001/2004)

*막대 하나로 캘리브레이션이 "언제" 가능한가 — 자유도 회계로 답한 논문. 원문(MSR-TR-2001-120)을 직접 읽고 정리*

Zhengyou Zhang, *Camera Calibration With One-Dimensional Objects*, MSR-TR-2001-120 (2001.12) → IEEE TPAMI 26(7), 2004. ([PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-120.pdf)) 평면 보드 캘리브레이션(TPAMI 2000)의 저자가 3D→2D→0D(자기캘리브레이션)로 채워져 있던 타깃 차원 스펙트럼에서 **1D의 빈칸**을 메운 논문이다.

![1d target](../_static/vision/fig_1d.svg)

## 한 줄 요약

> 자유롭게 움직이는 1D 물체로는 카메라 캘리브레이션이 **원리적으로 불가능**함을 자유도 회계로 증명하고, **한 점을 고정한 채 회전**하는 1D 물체(간격을 아는 공선점 3개)에 대해서는 **6회 이상 관측이면 폐형해**가 존재함을 보인 뒤, MLE 정제와 시뮬·실측 검증까지 완결한 "1D 캘리브레이션의 존재 정리".

## 문제 — 왜 하필 1D인가

원문 서론의 동기가 정확히 실무적이다: **멀리 떨어져 설치된 멀티카메라**를 캘리브레이션하려면 모든 카메라가 동시에 같은 타깃을 봐야 하는데, 앞뒤로 마주 보는 카메라 구성에서 3D/2D 장치는 사실상 불가능하다("천장에 매단 공 줄이면 된다" — 원문의 예). 1D 물체는 싸고, 휴대성이 좋고, 어느 방향에서도 같은 구조라 이 제약이 없다. 남는 질문은 정보량 — 막대 하나가 내부 파라미터 5개($\alpha, \beta, \gamma, u_0, v_0$)를 결정할 수 있는가?

## 방법

### 자유 이동은 안 된다 — 원문의 계수 회계 (§2.2)

카메라 좌표계를 물체 정의에 쓰면(R=I, t=0) 외부 파라미터가 사라지고 순수 회계만 남는다:

| 설정 | 미지수 | 방정식 | 판정 |
|---|---|---|---|
| 자유 이동, 점 2개(길이 기지) | $5 + 5N$ | $4N$ | 불가능 |
| 자유 이동, 공선점 3개 | $5 + 5N$ | $6N$이지만 **독립은 $5N$** | 불가능 |
| 자유 이동, 점 4개 이상 | $5+5N$ | 여전히 $5N$ | 불가능 |
| **한 점 고정 + 회전, 공선점 3개** | $8 + 2N$ | $2 + 3N$ | **$N \ge 6$이면 가능** |

세 번째 점이 방정식을 2개가 아니라 **1개만** 더하는 이유: 공선성은 투영에 보존되므로 세 이미지 점도 공선 — 관측이 서로 독립이 아니다. 네 번째 점부터는 **cross-ratio 불변성** 때문에 독립 방정식이 아예 늘지 않는다(정확도에는 도움 — 데이터 중복). 고정점 설정에서는 $A$의 3좌표가 전 프레임 공통이 되고 프레임당 미지수가 방향 2개($\theta, \phi$)로 줄어, $2+3N \ge 8+2N \Leftrightarrow N \ge 6$.

### 기본 제약식 유도 (§3.1) — 최소 구성(공선점 3개)

고정점 $A$, 자유끝 $B$($\|B-A\|=L$ 기지), 내분점 $C = \lambda_A A + \lambda_B B$($\lambda$들 기지). 관측 $\tilde{\mathbf a}, \tilde{\mathbf b}, \tilde{\mathbf c}$와 미지 깊이 $z_A, z_B, z_C$로 $A = z_A \mathbf{A}^{-1}\tilde{\mathbf a}$ 등으로 쓰고 내분식에 대입하면 $z_C\tilde{\mathbf c} = z_A\lambda_A\tilde{\mathbf a} + z_B\lambda_B\tilde{\mathbf b}$. 양변에 $\tilde{\mathbf c}$를 외적해 $z_C$를 소거하면

$$ z_B = -z_A\,\frac{\lambda_A(\tilde{\mathbf a}\times\tilde{\mathbf c})\cdot(\tilde{\mathbf b}\times\tilde{\mathbf c})}{\lambda_B(\tilde{\mathbf b}\times\tilde{\mathbf c})\cdot(\tilde{\mathbf b}\times\tilde{\mathbf c})} $$

이를 길이 제약에 넣으면 프레임당 하나의 스칼라 방정식이 남는다:

$$ z_A^2\,\mathbf{h}^\top \mathbf{A}^{-\top}\mathbf{A}^{-1}\,\mathbf{h} = L^2, \qquad
\mathbf{h} = \tilde{\mathbf a} + \frac{\lambda_A(\tilde{\mathbf a}\times\tilde{\mathbf c})\cdot(\tilde{\mathbf b}\times\tilde{\mathbf c})}{\lambda_B(\tilde{\mathbf b}\times\tilde{\mathbf c})\cdot(\tilde{\mathbf b}\times\tilde{\mathbf c})}\,\tilde{\mathbf b} $$

$\mathbf{h}$는 관측만으로 계산되고, $\mathbf{A}^{-\top}\mathbf{A}^{-1}$은 절대원뿔의 상(IAC) — Zhang 2000의 $\omega$와 같은 대상이 1D에서도 나타난다.

### 폐형해 (§3.2)

$\mathbf{B} = \mathbf{A}^{-\top}\mathbf{A}^{-1}$은 대칭이라 6-벡터 $\mathbf b$로 쓰고, $\mathbf x = z_A^2\mathbf b$로 두면 각 프레임이 $\mathbf v^\top\mathbf x = L^2$ 꼴의 **선형** 방정식이 된다($\mathbf v$는 $\mathbf h$ 성분의 2차 조합). $N$개를 쌓아 최소제곱 $\mathbf x = L^2(\mathbf V^\top\mathbf V)^{-1}\mathbf V^\top\mathbf 1$로 풀고, $\mathbf x$의 6성분에서 $v_0 \to z_A \to \alpha \to \beta \to \gamma \to u_0$ 순으로 닫힌 식이 전부 나온다. $z_A$가 복원되므로 $B, C$의 3D 위치까지 따라온다.

### MLE 정제 (§3.3)

폐형해는 대수적 거리라 물리적 의미가 없다 — 세 점 모두의 재투영 오차 합
$\sum_i (\|\mathbf a_i - \phi(\mathbf A, A)\|^2 + \|\mathbf b_i - \phi(\mathbf A, B_i)\|^2 + \|\mathbf c_i - \phi(\mathbf A, C_i)\|^2)$
을 LM(Minpack)으로 최소화한다. $B_i$는 구면각으로 $B = A + L[\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta]^\top$ 파라미터화 — 프레임당 2 미지수라는 회계가 구현에서도 유지된다(총 $8+2N$).

## 결과 — 원문 수치 그대로

- **시뮬레이션**: $\alpha=\beta=1000$, $(u_0,v_0)=(320,240)$, 640×480, 길이 70 cm 스틱, $A=[0,35,150]^\top$, 방향 100개 균일 샘플. 노이즈 0.1~1 px, 각 120회 시행. 오차는 노이즈에 거의 선형 증가 — 1 px 노이즈에서 폐형해 상대오차 ~12%, **비선형 정제 후 ~6%**(절반). 주점 오차는 Triggs의 제안대로 초점거리 대비 상대값으로 측정.
- **실측**: 아이 장난감 구슬 3개를 실로 꿴 스틱(간격 ~14 cm), 한쪽 끝을 **책으로 눌러 고정**하고 흔들며 150프레임 비디오. 구슬은 RGB 가우시안 블롭으로 검출. 같은 카메라를 평면 보드(Zhang 2000) 5장으로도 캘리브레이션해 비교 — 비선형해 기준 상대차 $\alpha$ 1.15%, $\beta$ 1.69%, $u_0$ 2.23%, $v_0$ 1.84%, skew 각도차 0.29°. **"약 2%"**.
- 오차 원인도 원문이 정직하게 나열한다: 고정점이 실제로는 표면에서 **미끄러졌고**, 구슬 간격은 자와 눈대중으로 쟀다. "이 조건에서 2%면 고무적"이라는 결론.

## 내 실습 연결 — 조선소의 1D 타깃

계측 과제 현장(외부 캘리브레이션)에서 1D 타깃을 쓴 이유가 논문의 동기와 정확히 겹친다:

- W.D 3.7~17.2 m에서 대형 화이트보드는 운반·거치가 곧 비용이다. 막대는 사람이 들고 서면 끝 — 원문이 말한 "동시 가시성" 이점의 지상 계측판.
- 스테레오 리그 **두 카메라가 동시에 관측** → 카메라 간 상대 포즈를 잇는 용도. 내부는 실험실에서 [ray calibration](ray-calibration.md)으로 이미 확보했으므로, 현장의 1D가 (이 논문이 다룬) 약한 내부 제약까지 떠맡을 필요가 없다 — 문제를 외부 추정으로 한정하면 1D의 약점이 사라진다.
- 3차 현장 계측: 화이트보드 평균 2.46 mm vs **1D 타깃 2.68 mm** — 정확도 동급을 실측으로 확인, 결론은 "타깃 선택은 정확도가 아니라 운용성의 문제".

면접식 한 줄: **"1D 캘리브레이션의 존재 조건(고정점 회전, N≥6)을 알면 현장에서 무엇을 포기해도 되는지 계산할 수 있다 — 우리는 내부를 실험실에서 끝냈기에 현장의 1D로 충분했다."**

## 한 줄 평 / 한계

"어떤 타깃이면 충분한가"를 계수 세기 하나로 끝까지 밀어붙인, 공학보다 수학에 가까운 논문. 실험 장치가 장난감 구슬과 책이라는 점이 오히려 이 논문의 성격을 잘 보여준다 — 기여는 장치가 아니라 **존재 증명**이다. 한계도 명확하다: 단일 카메라 정밀도는 2D 보드 대비 열세(자체 실험에서도 ~2% 차이), 점 검출·고정점 유지에 민감. 그러나 "멀리 떨어진 멀티카메라의 동시 가시성"이라는 원래 동기가 이후 모션캡처 wand 캘리브레이션 관행으로 실현됐다는 점에서, 작지만 오래 남는 결과다.
