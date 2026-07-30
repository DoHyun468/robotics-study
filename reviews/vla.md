# Vision-Language-Action (VLA)

언어 + 이미지를 받아 로봇 **action을 직접** 내는 정책 모델 리뷰. 인지·결정·제어를 하나의 정책이 통째로 대체하는 계열 — 우리 [컨텍스트 사다리](../context.md)의 L3에 해당한다.

- [OpenVLA (2024)](openvla.md) — Llama-2 7B + DINOv2/SigLIP 기반 오픈 VLA. LIBERO 4-suite를 직접 재현하고 LoRA 파인튜닝까지 돌렸다 → [VLA 실습 페이지](../vla.md).

## 우리 실측 요약 (LIBERO 4-suite, 20 ep/suite)

| suite | 우리 실측 | 논문 보고 |
|-------|-----------|-----------|
| spatial | 80% | 84.7% |
| object | 85% | 88.4% |
| goal | 85% | 79.2% |
| long | **45%** | 53.7% |

패턴 정합(long이 최난이도), 평균 74% ≈ 93M Octo(75.1%) — **LIBERO에선 7B의 파라미터 이점이 거의 안 드러난다**. 자체 LoRA 파인튜닝은 0.45 epoch bounded run에서 0%(norm stats 점검으로 undertrain 확인) — "체크포인트 재현"과 "훈련시켜 쓸 만하게"는 전혀 다른 난이도다.

## World model 계열과의 관계

VLA는 **모방(시연)** 신호로, [world-model RL](../world-models/latent.md)은 **보상** 신호로 배운다 — 문제 설정이 달라 같은 벤치로 직접 비교할 수 없다([Concepts](../concepts.md) 참고). 다만 출력단의 발상은 공유한다: OpenVLA의 action 256-bin 이산화는 [DreamerV3](../world-models/dreamerv3.md)의 twohot, [MuZero](../world-models/muzero.md)의 categorical support와 같은 "연속 회귀 → 이산 분포" 계열이다.

*향후: Octo · RT-2 · π0 등 추가 예정.*
