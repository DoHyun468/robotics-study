# Contact-GraspNet (2021)

> Sundermeyer, Mousavian, Triebel, Fox — *Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes*, ICRA 2021 (NVIDIA).

## 한 줄 요약
> 잡을 지점(contact)을 **관측된 3D 점 위에** 못박아, 6-DoF grasp 회귀를 관측 표면 위의 저차원 문제로 줄인다 — clutter에서 실시간 6-DoF grasp.

## 문제
6-DoF grasp을 depth/pointcloud에서 직접 뽑는 건 어렵다. grasp은 $SE(3)$의 6차원 연속 자세라 탐색공간이 크고, 선행 방법(6-DOF GraspNet 등)은 **샘플→평가**를 반복해 느리거나 완전한 물체 모델을 요구한다. clutter(어질러진 통) 안에서 충돌 없이, 물체 모델 없이, 빠르게 grasp을 내는 게 목표.

## 방법
핵심 통찰: **grasp의 접촉점은 카메라가 실제로 본 표면 점 위에 있어야 한다.** 관측 pointcloud의 한 점 $c$를 grasp의 한 접촉점으로 고정하면, 남는 자유도가 크게 줄어 per-point 예측으로 바뀐다.

각 점 $c_i$에 대해 PointNet++ 백본이 4가지를 예측한다:

- grasp 신뢰도 $s_i \in [0,1]$
- 접근 방향(approach) 단위벡터 $\mathbf{a}_i$
- baseline(그리퍼 개폐축) 단위벡터 $\mathbf{b}_i$
- grasp 폭 $w_i$

그러면 6-DoF 그리퍼 자세를 **닫힌 형태로 복원**한다(회전은 세 직교축, 위치는 접촉점에서 폭의 절반만큼 이동):

$$R_i = \big[\,\mathbf{b}_i,\ \ \mathbf{a}_i \times \mathbf{b}_i,\ \ \mathbf{a}_i\,\big],\qquad
\mathbf{t}_i = c_i + \tfrac{w_i}{2}\,\mathbf{b}_i - d\,\mathbf{a}_i$$

($d$=그리퍼 baseline↔접촉 오프셋 상수). 손실은 **접촉 신뢰도**(ACRONYM 시뮬 grasp이 그 점 근처에 있으면 positive)의 BCE + 복원된 그리퍼 **control-point 기하 손실**(ADD-S 류) + 폭 회귀. 학습은 전량 시뮬(ACRONYM), 추론 시 씬 pointcloud에서 collision 필터로 충돌 grasp을 제거한다.

## 결과
- 전량 시뮬 학습만으로 **실제 clutter에 zero-shot** 전이, 수백 ms 수준의 실시간.
- 우리 [bin picking A/B](../grasp_sota.md)에서도 실행: walled bin **33%** / open tray 20% (2020→2025 SOTA 6종 중 walled 최고였지만, 씬에 co-tune된 top-down 휴리스틱 63%는 못 넘음 — "논문 grasp score ≠ 이 그리퍼로 실제 성공").

## 내 실습 연결
이 논문의 **"접촉점을 관측 표면 점에 고정"** 아이디어가 우리 grasp 아크의 뼈대와 같다. [§11.9 접촉 제약 리타게팅](../human_pose.md)에서 로봇 손끝을 mug 표면 최근접점 + 법선으로 앵커링한 게 정확히 이 발상의 손 버전이고, [§11.10 grasp 합성](../human_pose.md)의 alternating projection은 "매 라운드 손끝을 도달 가능한 표면 점으로 재투영"한다. 차이: Contact-GraspNet은 **평행 그리퍼(2접촉)** 를 학습으로 한 번에 예측, 우리는 **4지 Allegro(다접촉)** 를 force-closure까지 최적화(→ [DexGraspNet](dexgraspnet.md) 계열).

## 한 줄 평 / 한계
접촉점을 관측에 못박은 표현이 우아하고 실용적이다. 다만 **평행 그리퍼 전제**(다지 손엔 그대로 못 씀)이고, 학습이 sim grasp label 품질에 종속되며, 우리 A/B가 보여주듯 **grasp 검출 점수와 실제 실행 성공 사이엔 그리퍼 co-tuning·제어라는 층**이 따로 있다.
