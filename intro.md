# Robotics Study

<span class="rs-tag">Spatial Intelligence → Embodied</span>

<p class="rs-lede">
카메라가 본 것을 로봇 손이 닿게 — <strong>perception → action</strong>. KAIST 기계·전전 배경의 3D perception(stereo, LiDAR, calibration)과 실무의 NeRF·3DGS·reconstruction 경험을 로봇 문제로 확장하며, 그 위에 학습(grasp SOTA, VLA)을 <strong>정직하게 A/B</strong> 한 실습·스터디 노트. 모든 수치는 실측이고, 안 되는 것은 안 된다고 적는다.
</p>

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Perception
:link: perception
:link-type: doc
캘리브레이션 → ICP 8단계. end-to-end **19mm**, 병목은 제어가 아니라 depth → stereo로 **−92%**.
:::

:::{grid-item-card} Manipulation
:link: manipulation
:link-type: doc
물성 4축 7+태스크. stack 95% · push 5/5 · pour ✗(정직한 실패) · rope 자유단 16mm.
:::

:::{grid-item-card} Grasp SOTA A/B
:link: grasp_sota
:link-type: doc
학습 grasp 6모델 vs heuristic. 전부 heuristic 아래 — 도메인갭·제어가 실제 레버.
:::

:::{grid-item-card} VLA
:link: vla
:link-type: doc
OpenVLA LIBERO 4-suite 재현(논문 정합) + 자체 LoRA 파인튜닝의 정직한 undertrain.
:::

:::{grid-item-card} Context
:link: context
:link-type: doc
"이걸 집어라"를 전달하는 3단계 — 포인터 → 언어(OWL-ViT) → VLA.
:::

:::{grid-item-card} Concepts
:link: concepts
:link-type: doc
RL vs world-model 정리. 왜 이 스터디가 perception→action에 집중하는가.
:::

:::{grid-item-card} World Models
:link: world-models/index
:link-type: doc
Dreamer · PlaNet · MuZero · V-JEPA-2 · 생성형 WM — arXiv 원문 대조 심화 리뷰.
:::

::::

<p class="rs-foot">
스택: MuJoCo(Franka Panda) · WSL2/RTX 4090 · 고전 파이프라인(기하 perception + DLS IK + PD 제어) + 학습(GraspNet · OpenVLA · …). 논문 리뷰·World Models는 사이드바 **Paper Reviews** 참조. 개인 스터디 — 회사와 무관.
</p>
