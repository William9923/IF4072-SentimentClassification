setup:
	sh ./setup.sh

format:
	python -m black .

complexity:
	radon cc src main.py -a

exp:
	sh ./run-exp.sh