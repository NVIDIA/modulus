install:
	pip install --upgrade pip && \
		pip install -e .

get-data:
	mkdir /data && \
		mkdir /data/nfs/ && \
		git clone https://gitlab-master.nvidia.com/modulus/modulus-data.git /data/nfs/modulus-data

black: 
	black --check ./

interrogate:
	cd modulus && \
                interrogate --ignore-init-method \
                --ignore-init-module \
                --ignore-module \
                --ignore-private \
                --ignore-semiprivate \
                --ignore-magic \
                --fail-under 99 \
                --exclude ["internal"] \
                --ignore-regex forward \
                --ignore-regex backward \
                --ignore-regex reset_parameters \
                --ignore-regex extra_repr \
                --ignore-regex MetaData \
                --ignore-regex apply_activation \
                --ignore-regex exec_activation \
                -vv \
                --color \
                . && \
		cd ../

license: 
	python test/ci_tests/header_check.py


doctest:
	coverage run \
                --rcfile='test/coverage.docstring.rc' \
                -m pytest \
                --doctest-modules modulus/ --ignore-glob=*internal*

pytest: 
	coverage run \
                --rcfile='test/coverage.pytest.rc' \
                -m pytest 

pytest-internal:
	cd test/internal && \
                pytest && \
		cd ../../

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --omit=*internal* --fail-under=80 && \
		coverage html
