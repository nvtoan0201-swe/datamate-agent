# Lệnh phát triển. Dữ liệu thật không có trong repo: đặt DATAMATE_DEV_SPLIT trỏ tới split có answers.jsonl.
.PHONY: help sync lint test eval eval-synth smoke check serve ui up report

DEV_SPLIT ?= $(DATAMATE_DEV_SPLIT)
PER_SHOP ?= 5
WORKERS ?= 8
EVAL_ARGS ?=

help:
	@echo "make sync        cài môi trường (uv sync)"
	@echo "make lint        ruff check"
	@echo "make test        pytest (shop synthetic + LLM giả)"
	@echo "make eval        eval trên dữ liệu thật: DATAMATE_DEV_SPLIT=... PER_SHOP=$(PER_SHOP) (cần LLM_BASE_URL/LLM_MODEL)"
	@echo "                 lưu eval/results/<timestamp>.json, so với baseline gần nhất, tụt quá ngưỡng thì fail"
	@echo "make report RUN=outputs/eval/<timestamp>   báo cáo lỗi theo nhóm nguyên nhân (cần DATAMATE_DEV_SPLIT)"
	@echo "make eval-synth  eval synthetic qua scripts/run.py với LLM giả (như CI)"
	@echo "make smoke       chạy entrypoint thi docker/run.sh với LLM giả"
	@echo "make check       lint + test + eval-synth + smoke"
	@echo "make serve       API online (uvicorn, cổng 8000; cần LLM_BASE_URL/LLM_MODEL)"
	@echo "make ui          UI Streamlit (cổng 8501) gọi API ở DATAMATE_API_URL"
	@echo "make up          docker compose up --build (API + UI, đọc .env)"

sync:
	uv sync

lint:
	uv run ruff check

test:
	uv run pytest -q

eval:
	@test -n "$(DEV_SPLIT)" || (echo "Đặt DATAMATE_DEV_SPLIT (hoặc DEV_SPLIT=...) trỏ tới split có answers.jsonl"; exit 2)
	uv run python scripts/eval.py --split "$(DEV_SPLIT)" --per-shop $(PER_SHOP) --workers $(WORKERS) $(EVAL_ARGS)

report:
	@test -n "$(RUN)" || (echo "Đặt RUN=<thư mục chạy có submission.jsonl và traces/>"; exit 2)
	uv run python scripts/error_report.py --split "$(DEV_SPLIT)" --run "$(RUN)"

eval-synth:
	uv run python scripts/eval.py --synthetic

smoke:
	bash scripts/smoke_entrypoint.sh

check: lint test eval-synth smoke

serve:
	uv run --extra service uvicorn --factory datamate.service.api:create_app --port 8000

ui:
	uv run --extra ui streamlit run src/datamate/service/ui.py

up:
	docker compose up --build
