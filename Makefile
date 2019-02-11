init:
	pipenv install --three --dev
	pipenv run python -m ipykernel install --user --name=spatial-assocr

tests:
	pipenv run pytest
