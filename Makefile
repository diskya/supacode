PY=python
PIP=pip

.PHONY: bootstrap index query chat lint

bootstrap:
	$(PIP) install -r requirements.txt || true

index:
	$(PY) -m codexbot.cli index --repo $(REPO) --index-dir $(INDEX_DIR) --embedder $(EMBEDDER)

query:
	$(PY) -m codexbot.cli query --index-dir $(INDEX_DIR) --q "$(Q)" --top-k $(TOPK)

chat:
	$(PY) -m codexbot.cli chat --index-dir $(INDEX_DIR) --q "$(Q)" --top-k $(TOPK) --dry-run

# Defaults
INDEX_DIR?=.codexbot/index
EMBEDDER?=hash
TOPK?=10
