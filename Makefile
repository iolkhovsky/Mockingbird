.PHONY: install
install:
	pip3 install -r requirements.txt

.PHONY: dataset
dataset:
	python3 preprocessor.py \
		--telegram_data=raw_data/result.json \
		--actor_to_alias=raw_data/actor_to_alias.json \
		--output=ift_dataset \
		--cooling_time=30 \
		--max_replicas=10 \
		--min_replicas=3 \

.PHONY: finetune
finetune:
	python3 finetune.py \
		--dataset=ift_dataset \
		--checkpoint=mistralai/Mistral-7B-v0.1 \
		--ft_checkpoint=ft_ckpt \
