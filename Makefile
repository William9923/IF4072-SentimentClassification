setup:
	sh ./setup.sh

format:
	python -m black .

complexity:
	radon cc src main.py -a