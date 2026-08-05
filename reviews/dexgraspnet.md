# DexGraspNet (2023)

> Wang, Liu, Chen, ... , Wang — *DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation*, ICRA 2023 (PKU EPIC · Stanford).

## 한 줄 요약
> **미분가능 force-closure 에너지**로 다지 손 grasp을 자동 합성해, 5355개 물체·132만 grasp의 대규모 dexterous 데이터셋을 시뮬로 만든다.

## 문제
다지(dexterous) 손 학습엔 데이터가 필요한데, 실제 다지 grasp 데이터셋은 작고 다양성이 낮다. 사람이 일일이 만들 수도 없다. **물체 수천 개에 대해 물리적으로 유효한(안 미끄러지고, 안 뚫고, 관절限 지키는) 다지 grasp을 대량 자동 생성**하는 게 목표.

## 방법
grasp을 **에너지 최소화**로 합성한다. Shadow Hand 자세(손 global pose + 관절각) $q$에 대해 에너지:

$$E(q) = E_{\text{fc}} \;+\; w_{\text{dis}}\,E_{\text{dis}} \;+\; w_{\text{pen}}\,E_{\text{pen}} \;+\; w_{\text{spen}}\,E_{\text{spen}} \;+\; w_{\text{joints}}\,E_{\text{joints}}$$

- $E_{\text{dis}}$: 선택된 손끝(접촉 후보)과 물체 표면 사이 거리 → 접촉점을 표면에 붙임
- $E_{\text{pen}}$ / $E_{\text{spen}}$: 물체 관통 / 자기 관통 패널티
- $E_{\text{joints}}$: 관절限 위반 패널티
- $E_{\text{fc}}$: **미분가능 force-closure**(DFC, Liu et al. 2021) — 접촉점 $x_i$와 법선으로 grasp matrix $G=[\,\dots,\ (I;\ [x_i-p]_\times)\,n_i,\ \dots]$를 만들고, 접촉 wrench들이 원점을 감싸는(=임의 외력을 버티는) 조건을 완화해 미분가능 스칼라로 잰다:

$$E_{\text{fc}} = \big\lVert G\,\mathbf{1} \big\rVert^2 \;+\; \text{(wrench span term)}$$

이 에너지를 MALA(Metropolis-adjusted Langevin) 류 샘플링으로 최소화 → 다양한 유효 grasp을 대량 생성하고, Isaac Gym에서 물리 시뮬로 성공한 것만 남긴다.

## 결과
- **1.32M grasp / 5355 물체 / Shadow Hand**, 물체당 수백 grasp의 다양성. 종전 다지 데이터셋 대비 규모·다양성 압도.
- 이 데이터로 학습한 grasp 생성 모델이 미보지 물체에 일반화.

## 내 실습 연결
DexGraspNet은 우리 [§11.10–11.12 grasp 합성](../human_pose.md)의 **대규모·정공법 버전**이다. 우리가 축소판으로 직접 구현한 것들이 그대로 대응한다:

| 우리 grasp 아크 | DexGraspNet |
|---|---|
| [§11.10](../human_pose.md) alternating projection으로 손끝→표면 | $E_{\text{dis}}$ (표면 거리 항) |
| [§11.10](../human_pose.md) 자기충돌 패널티 + [§11.9](../human_pose.md) 관통 | $E_{\text{spen}}, E_{\text{pen}}$ |
| [§11.11](../human_pose.md) Ferrari–Canny force-closure로 후보 선별 | $E_{\text{fc}}$ (미분가능 force-closure) |
| [§11.11](../human_pose.md) "gap이 아니라 ε를 최적화하라" | 에너지에 **force-closure 항을 명시**하는 이유 그 자체 |

특히 우리 [§11.11의 반례](../human_pose.md)(접촉 거리만 줄이면 오히려 불안정)가 DexGraspNet이 왜 $E_{\text{dis}}$만이 아니라 $E_{\text{fc}}$를 반드시 넣는지를 정확히 설명한다. 데이터셋 물체 mesh는 [§11.7](../human_pose.md)에서 우리가 쓴 BOP mesh + SDF 접촉과 같은 계열.

## 한 줄 평 / 한계
"grasp을 **합성 문제**로 보고 물리 에너지로 대량 생성"이라는 방향이 우리 실습과 개념적으로 완전히 일치해서, 우리가 손으로 만든 조각들의 정답지 역할을 한다. 한계: 전량 시뮬 합성이라 **sim-to-real gap**(표면 마찰·물성·센싱 노이즈)이 남고, force-closure는 정적 안정성일 뿐 **실제 파지 성공(동적·접촉 물리)** 과는 또 다른 층이다 — 우리 [bin picking A/B](../grasp_sota.md)가 보여준 "점수 ≠ 실행 성공"과 같은 경고.
