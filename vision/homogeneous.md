# 2. Homogeneous Coordinates

> 동차좌표는 "평행이동과 투영도 행렬곱으로 쓰겠다"는 표기 계약이다. 계약의 대가는 스케일 동치 — $(x,y,1)$과 $(2x,2y,2)$는 같은 점이다.

## 정의

2D 점 $(x,y)$를 $(x,y,1)$로, 일반적으로 $w\neq 0$인 $(wx, wy, w)$ 전부와 동일시한다:

$$ (x, y) \;\longleftrightarrow\; \lambda\,(x, y, 1),\quad \lambda \neq 0 $$

## 왜 쓰는가

1. **평행이동이 선형이 된다.** $\mathbf{x}' = \mathbf{x} + \mathbf{t}$는 비선형(affine)이지만, 동차좌표에선
   $\begin{bmatrix} I & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$ 곱 하나다. 변환 합성이 전부 행렬곱으로 통일된다.
2. **투영이 선형이 된다.** 핀홀 투영의 $/Z$ 나눗셈을 "마지막에 $w$로 나눈다"로 미루면, 투영 자체는 $3\times 4$ 행렬 $P = K[R|\mathbf{t}]$ 곱이 된다.
3. **무한원점을 다룰 수 있다.** $w=0$인 $(x,y,0)$은 방향만 있는 점 — 평행선의 교점(소실점)이 유한한 대상으로 들어온다.

## 직선도 같은 언어로

2D 직선 $ax+by+c=0$은 $\mathbf{l}=(a,b,c)^\top$로 쓰고, 점 $\mathbf{x}$가 직선 위에 있음은 $\mathbf{l}^\top \mathbf{x} = 0$. 두 점의 외적이 직선, 두 직선의 외적이 교점이 된다:

$$ \mathbf{l} = \mathbf{x}_1 \times \mathbf{x}_2, \qquad \mathbf{x} = \mathbf{l}_1 \times \mathbf{l}_2 $$

Epipolar 기하의 $\mathbf{l}' = F\mathbf{x}$가 자연스럽게 읽히는 것도 이 표기 덕분이다.

## 실전 메모

- 수치 안정성 때문에 마지막 성분을 1로 정규화하는 시점을 통일하라. 최적화 중에는 정규화를 미루는 편이 낫다.
- "$\sim$" (스케일 동치) 기호를 등호처럼 쓰다 스케일을 잃는 버그가 흔하다 — 특히 $E$, $F$, $H$ 추정 후 복원 단계에서.
