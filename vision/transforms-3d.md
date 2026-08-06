# 5. 3D 변환

> 3D 강체 변환 $T=(R,\mathbf{t}) \in SE(3)$. 문제는 표현이다 — 회전을 어떻게 쓸 것인가가 실무 버그의 절반을 만든다.

## 회전의 표현들

| 표현 | 파라미터 | 강점 | 함정 |
|---|---|---|---|
| 회전행렬 $R\in SO(3)$ | 9 (제약 6) | 합성·적용이 곧 행렬곱 | 수치 누적 시 직교성 깨짐 → 재직교화 필요 |
| 축-각 $(\hat{\mathbf{k}},\theta)$ / 회전벡터 | 3 | 최소 표현, 최적화 친화 (Rodrigues) | $\theta=\pi$ 부근 모호 |
| 오일러각 | 3 | 사람이 읽기 좋음 | **짐벌락**, 축 순서 12가지 혼란 |
| 쿼터니언 $q$ | 4 (단위) | 보간(slerp)·합성 안정 | $q$와 $-q$ 동일(double cover), 정규화 관리 |

Rodrigues 공식으로 축-각 ↔ 행렬을 오간다:

$$ R = I + \sin\theta\,[\hat{\mathbf{k}}]_\times + (1-\cos\theta)\,[\hat{\mathbf{k}}]_\times^2 $$

## SE(3)와 합성

$$ T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix},\qquad
T_{A\to C} = T_{B\to C}\,T_{A\to B} $$

읽는 방향을 고정하라(위 표기: "A좌표를 B좌표로"). TF 트리(ROS)가 하는 일이 정확히 이 합성의 관리다.

## 실전 함정

- **오일러각으로 오차를 보고하지 말 것** — 축 순서에 따라 값이 달라진다. 자세 오차는 geodesic angle $\theta = \arccos\frac{\mathrm{tr}(R_{gt}^\top R)-1}{2}$ 하나로.
- 보간은 쿼터니언 slerp. 행렬 성분 선형보간은 회전이 아니게 된다.
- 라이브러리 간 쿼터니언 순서($wxyz$ vs $xyzw$)가 다르다 — hand-eye 구현에서 실제로 겪은 버그.

## 내 실측 연결

hand-eye AX=XB([리뷰 허브](../reviews/calibration.md) 참고)의 해법들(Tsai-Lenz, Daniilidis)은 전부 "회전을 어떤 표현으로 풀 것인가"의 변주다. 내 실측 4.2 mm / **0.10°**의 자세 오차도 geodesic angle로 잰 값이다.
