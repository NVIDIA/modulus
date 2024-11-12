lock_deps:
	pip freeze | grep -v @ > requirements.txt

install:
	pip install -r requirements.txt


LINT_FILES = run_inference.py score_inference.py utils/diffusions tests/

lint:
	black $(LINT_FILES)
	ruff $(LINT_FILES)


test:
	pytest tests/
	python3 run_inference.py
	python3 score_inference.py test_images/


download_dev_data:
	mkdir -p data
	rsync -avP dtn02.nersc.gov:/global/cfs/cdirs/m4331/hrrr_v3_subset data