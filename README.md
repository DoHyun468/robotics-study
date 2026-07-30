# Robotics Study — Spatial Intelligence → Embodied

카메라 perception → 로봇 action 실습 + grasp SOTA / VLA / world-model A/B + 논문 리뷰. 개인 스터디·포트폴리오. (Classic **Jupyter Book**, GitHub Pages 호스팅.)

## 로컬 빌드
```bash
pip install -r requirements.txt          # jupyter-book==0.15.1 (classic, NOT >=1.0)
jupyter-book build .                      # → _build/html/index.html
python -m http.server 8799 -d _build/html # 미리보기: http://127.0.0.1:8799
```

## 배포 (GitHub Pages)
`main`에 push하면 `.github/workflows/deploy.yml`이 빌드 후 Pages에 자동 게시.
저장소 Settings → Pages → **Source: GitHub Actions** 한 번만 설정.

## 구조
- `_config.yml` / `_toc.yml` — 책 설정·목차
- `*.md` — 본문 (perception · manipulation · grasp_sota · vla · context · concepts)
- `reviews/` — 논문 리뷰 (+ `_template.md`)
- `_static/` — 데모 영상·이미지

모든 수치는 실측이며, 실패는 실패로 기록한다.
