activate:
	poetry shell

install:
	poetry install
	pip install pre-commit
	pre-commit install

lint:
	pre-commit run --all-files

setup-ipykernel:
	poetry shell; \
	pip install ipykernel; \
	python -m ipykernel install --user --name jax_center_loss
	