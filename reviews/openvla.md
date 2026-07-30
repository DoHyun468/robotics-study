# OpenVLA

## 한 줄 요약
> Llama-2 7B + DINOv2/SigLIP 비전 백본 위에 로봇 action을 **토큰**으로 얹어 학습한 **오픈 VLA** — 이미지+언어 → 7-DoF action. Open X-Embodiment 대규모 모방학습으로 사전학습하고, downstream은 LoRA로 파인튜닝. 우리가 LIBERO 4-suite를 직접 재현하고 자체 파인튜닝까지 돌린 모델.

## 문제
로봇 조작 정책은 보통 태스크·로봇(embodiment)마다 새로 학습된다 — 데이터·재현비용이 크고 일반화가 약하다. VLM(비전-언어 모델)의 인터넷급 사전학습과 언어 일반화를 로봇 action에 이식하면, **하나의 범용·언어조건 정책**으로 여러 태스크를 커버할 수 있지 않을까? 그리고 그것을 **오픈 가중치**로 풀어 누구나 파인튜닝하게 하자는 것이 OpenVLA의 문제의식이다.

## 방법
- **백본**: Prismatic VLM = 비전 인코더 2종(DINOv2 + SigLIP)으로 이미지를 임베딩하고 Llama-2 7B가 언어와 함께 처리.
- **action = 토큰**: 연속 7-DoF action(Δposition·Δrotation·gripper)을 이산 bin으로 양자화해 언어 토크나이저의 토큰에 매핑 → **다음-토큰 예측(behavior cloning)** 으로 학습. 정책 학습이 곧 language modeling이 된다.
- **사전학습**: Open X-Embodiment(~1M 로봇 궤적, 다양한 로봇·태스크).
- **downstream**: 전체 파인튜닝 대신 **LoRA**로 특정 벤치(예: LIBERO 4-suite)에 적응. action normalization 통계(norm stats)를 데이터셋별로 맞춰줘야 디코딩이 정상 동작.

## 결과 (우리 재현)
LIBERO 4-suite를 20-ep씩 직접 돌려: **spatial 80 / object 85 / goal 85 / long 45** — 논문(84.7 / 88.4 / 79.2 / 53.7)과 **패턴 정합**(spatial~goal 높고 long이 최난이도). 평균 74%로, 93M짜리 Octo(75.1%)와 거의 같다 → **7B의 20배 파라미터가 LIBERO에선 거의 안 드러남**. 상세·환경 셋업(WSL2 `ov` env, protobuf 삼각충돌·flash-attn·robosuite evdev 우회 등)은 [VLA 페이지](../vla.md).

## 내 실습 연결
우리 [컨텍스트 사다리](../context.md)의 **L3**(언어+이미지→action, 인지·결정·IK를 정책이 통째로 대체)에 해당. 나아가 자체 **LoRA 파인튜닝 파이프라인**(RLDS→train→merge→eval)을 end-to-end 구동했다. 단 GPU 시간 제약으로 돌린 bounded run(**1500 step ≈ 0.45 epoch**)은 LIBERO-Spatial **0%** — norm stats를 직접 점검해 버그가 아니라 순수 **undertrain**임을 확인했다. 즉 "체크포인트를 실행해 논문값 재현"과 "VLA를 훈련시켜 쓸 만하게 만들기"는 전혀 다른 난이도라는 걸 실측했다(현재 lr5e-4로 다중 epoch 장기 run 진행 중).

## 한 줄 평 / 한계
오픈 가중치 + LoRA 파인튜닝 용이 = **접근성 최고**의 VLA. 단 (1) 7B라 추론·파인튜닝 compute가 크고(단일 GPU로 수렴시키려면 다중 epoch·장시간), (2) 반응형이라 명시적 계획이 없으며, (3) LIBERO에서 93M Octo와 성능차가 미미 = **스케일이 항상 답은 아니다**. 그럼에도 "VLM을 로봇 정책으로"의 재현 가능한 오픈 기준선으로서 가치가 크다.
