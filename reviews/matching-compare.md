# SuperGlue vs LoFTR vs DKM — 무엇이 다르고 언제 무엇을 쓰나

*세 매처를 전부 실사용해 본 입장에서 정리하는 비교 — "키포인트"라는 말이 세 논문에서 서로 다른 것을 가리킨다는 사실이 이 페이지의 출발점이다. 수치는 각 리뷰에서 원문 확인한 값.*

## 한눈 비교

| | [SuperGlue](superglue.md) (2020) | [LoFTR](loftr.md) (2021) | [DKM](dkm.md) (2023) |
|---|---|---|---|
| 계열 | sparse (detector 기반) | semi-dense (detector-free) | dense |
| 매칭 대상 | SuperPoint 검출점 (수백~2k) | coarse 그리드 셀 (1/8 해상도 전체) | **모든 픽셀** + certainty |
| 핵심 기제 | attentional GNN + Sinkhorn OT | Linear Transformer + dual-softmax, coarse→fine | GP 전역 매처 + warp refinement |
| 서브픽셀 | 검출점 위치 그대로 | fine 단계 heatmap 기대값 | dense warp 회귀 |
| ScanNet AUC@5° | 16.16 | 22.06 | **29.4** |
| MegaDepth AUC@5° | 42.2 (SP+SG) | 52.8 | **60.4** |
| 속도(대략) | **69 ms** (GTX1080) | 116 ms (2080Ti) | 수백 ms~ (가장 무거움) |
| 신뢰도 신호 | assignment 확률 $P_{ij}$ (Sinkhorn) | confidence matrix $\mathcal{P}_c$ (dual-softmax) | certainty 맵 (전용 branch) |
| 검출기 의존 | **있음** (SuperPoint 상한) | 없음 | 없음 |
| 주 실패 모드 | 저텍스처에서 검출 자체 실패 | 반복 구조에서 전역 attention 혼동 | 비용, 극단 시점차(→RoMa로 개선) |

## "키포인트"의 정체가 셋 다 다르다

같은 단어를 쓰지만 가리키는 대상이 다르고, 이 차이가 실패 모드를 결정한다.

- **SuperGlue의 keypoint = 검출점.** SuperPoint가 "코너스러움"을 학습해 찍은 점이다. 반복성(같은 물리점이 두 뷰에서 다 검출될 것)이 전제 — 매끈한 강판·벽처럼 **검출할 코너가 없는 표면에서는 매칭 이전에 게임이 끝난다.** 대신 점이 적어 빠르고, descriptor를 캐시할 수 있어 SfM(COLMAP)과 궁합이 좋다.
- **LoFTR의 매치 후보 = coarse 그리드 셀 중심.** 검출을 건너뛰고 1/8 해상도 feature map의 모든 위치가 후보다 — **저텍스처에도 후보가 "존재는 한다"**는 것이 결정적 차이. 위치 정밀도는 fine 단계의 기대값 회귀가 만든다. 대가는 전역 attention이 비슷한 구조를 혼동하는 것(반복 패턴에서 원거리 오매칭).
- **DKM의 매칭 = 픽셀 전부 + certainty.** 아예 dense warp를 회귀하고, 어디를 믿을지는 **certainty 맵**이 말해준다. 뽑아 쓸 때는 certainty 가중 + KDE 역수 balanced sampling. "keypoint를 고르는 문제"가 "신뢰도를 거르는 문제"로 바뀐다 — 필터링 파이프라인과 궁합이 가장 좋다.
- (번외) **[COTR](cotr.md)·[CAPS](caps.md)의 질의 좌표.** "네가 궁금한 좌표를 물어라"는 함수형 — 계측점처럼 **위치가 미리 정해진 점**의 대응에는 이 축이 정확히 맞는다. 내 석사 파이프라인의 최종 매칭기가 COTR이었던 이유.

## 각자의 장단 — 실사용 감각

**SuperGlue** — 장점: 빠르고(69 ms), 희소 출력이라 후단(RANSAC·BA)이 가볍고, 부분 매칭 개념(dustbin)이 명시적. **신뢰도의 정체**: Sinkhorn이 수렴시킨 soft assignment 행렬 $P$의 원소 자체가 매치 확률이다 — dustbin까지 포함한 합이 1로 정규화된 진짜 확률이라, 임계값 하나로 매치를 거르면 그게 곧 confidence 필터링이다(별도 헤드가 없다는 게 우아한 점). 단점: 상한이 검출기다 — SuperPoint가 못 찍으면 GNN이 아무리 좋아도 소용없다. 우리 강판 데이터에서 정확히 이 벽을 만났다.

**LoFTR** — 장점: 저텍스처·모션블러에서 후보가 살아 있다(detector-free의 존재 이유). Linear attention으로 semi-dense를 실용 속도에. **신뢰도의 정체**: coarse 단계에서 유사도 행렬에 **양방향 softmax를 곱한 dual-softmax confidence** $\mathcal{P}_c(i,j)$ — "i가 j를 고르고 j도 i를 고를" 결합 확률이라 상호 최근접(MNN)과 임계값($\theta_c{=}0.2$)이 자연스럽게 결합된다. fine 단계에는 별도로 heatmap 기대값의 **분산**이 나와 위치 불확실성으로 쓰인다(분산 가중 손실). 단점: 반복 구조(선체 보강재의 주기 패턴!)에서 coarse 전역 매칭이 헷갈리면 fine이 구제 못 한다 — 우리 데이터의 주 오매칭원. 그리고 매 쌍마다 전체를 다시 계산(캐시 불가).

**DKM** — 장점: 정확도 자체는 셋 중 최강(MegaDepth AUC@5° 60.4), certainty가 후단 필터와 자연 결합, dense라 임의 지점 대응을 보간으로 뽑을 수 있다. **신뢰도의 정체**: 앞의 둘과 달리 매칭 분포에서 파생되지 않고 **전용 branch가 픽셀별 certainty를 직접 회귀**한다 — GT는 "예측 warp가 depth 일관성 검사를 통과하는 픽셀"이라는 binary 라벨로 감독(즉 '맞출 수 있는 곳인가'를 배운다). 샘플링 때는 이 certainty에 KDE 역수를 곱해 공간적으로 고르게 뽑는다(balanced sampling, AUC +2.0). 단점: 가장 무겁고, 학습 분포 밖 극단 시점차는 여전히 어렵다(이 축을 밀어붙인 후속이 [RoMa](roma.md)).

**왜 앙상블이었나** — 석사 파이프라인에서 셋을 함께 쓴 이유는 성능 합산이 아니라 **실패 모드가 상보적**이어서다: 검출 실패(SG) / 전역 혼동(LoFTR) / 신뢰도 저하(DKM)가 서로 다른 프레임에서 터진다. 앙상블 keypoint → 신뢰도·epipolar·재투영 3단 필터 → ROI/guided mask로 탐색 공간 축소 → COTR 정밀 질의가 최종 구조(29.1→2.23 px).

## 언제 무엇을 쓰나 — 결정 규칙

| 상황 | 선택 | 이유 |
|---|---|---|
| SfM/SLAM 파이프라인, 실시간·대량 | SuperGlue → 지금은 [LightGlue](lightglue.md) | 희소·캐시·속도, COLMAP 호환 |
| 저텍스처·야간·블러가 주적 | LoFTR → 지금은 [Efficient LoFTR](eloftr.md) | 검출 실패 자체를 우회 |
| 정확도 최우선, 오프라인, 극단 시점차 | DKM → 지금은 [RoMa](roma.md) | dense+robust, certainty 필터 |
| 지정된 점(계측점·마커)의 대응 | COTR (+탐색 영역 제약) | 질의 기반이 문제 정의와 일치 |
| 포즈만 필요하고 이미지가 아주 다름 | [MASt3R](mast3r.md) | 매칭을 3D에 접지 — 극단 baseline 최강 |

한 줄 요약: **"어디를 매칭할 것인가"의 결정권이 검출기 → 그리드 → 신뢰도 맵으로 이동해 온 역사이고, 내 도메인(저텍스처·반복 패턴·지정 계측점)에서는 그 어느 하나도 단독으로 충분하지 않았다 — 그래서 파이프라인이 논문이 됐다.**
