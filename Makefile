.PHONY: install train test lint

# Instalar dependencias
install:
	pip install -r requirements.txt

# Ejecutar el pipeline completo (entrenar el modelo)
train:
	python src/train.py

# Ejecutar pruebas básicas de validación (puedes usar un framework como pytest)
test:
	pytest tests/

# Linting de código (para asegurarse de que el código siga buenas prácticas)
lint:
	flake8 src/
