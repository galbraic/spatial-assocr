init:
	rm -rf .venv
	python3 -m venv .venv
	source .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements-dev.txt; \
	pip install -e .; \
	cd kde/pyemd; \
    python setup.py install; \
	cd ../..; \
	ipython kernel install --user --name=spatial-assocr;

tests:
	source .venv/bin/activate; \
	pytest -vv
