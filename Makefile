setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python3 get_model.py

run_triton_server:
	sudo docker run --name container-triton --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
		-v/home/tari/Desktop/STUDY/Optimization/model_repository:/models \
		nvcr.io/nvidia/tritonserver:22.05-py3 tritonserver --model-repository=/models

stop_triton_server:
	sudo docker stop container-triton