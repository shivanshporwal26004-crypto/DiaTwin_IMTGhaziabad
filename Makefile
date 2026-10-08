.PHONY: install train app test diagram
install:
	pip install -r requirements.txt
train:           ## regenerate data, retrain models, rebuild metrics + figures (~2 min)
	python -m src.train
app:             ## launch the clinician dashboard
	streamlit run app/dashboard.py
test:
	pytest -q
diagram:
	python docs/make_architecture.py
