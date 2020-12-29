SAVE_DIR=/hdd1/minbyul/output/ner_output/
DATA_DIR=/home/minbyul/github/biobert-pytorch/datasets/

MAX_LENGTH=128
BERT_MODEL=bert-base-cased
OUTPUT_DIR=germeval-model
BATCH_SIZE=1
NUM_EPOCHS=35
SAVE_STEPS=-1
SEED=1

ENTITY=NCBI-disease
cuda=1
ADVERSARIAL=hotflip

bioner-adversarial:
	CUDA_VISIBLE_DEVICES=$(cuda) python run_ner.py \
		--data_dir $(DATA_DIR)/NER/$(ENTITY) \
		--labels $(DATA_DIR)/NER/$(ENTITY)/labels.txt \
		--model_name_or_path $(SAVE_DIR)/biobert/$(ENTITY)-softmax-$(NUM_EPOCHS)-removedot \
		--output_dir $(SAVE_DIR)/biobert/$(ENTITY)-softmax-$(NUM_EPOCHS)-$(ADVERSARIAL) \
		--max_seq_length  $(MAX_LENGTH) \
		--num_train_epochs $(NUM_EPOCHS) \
		--per_device_eval_batch_size $(BATCH_SIZE) \
		--save_steps $(SAVE_STEPS) \
		--seed $(SEED) \
		--do_eval \
		--do_predict \
		--wandb_name $(run_name) \
		--adversarial $(ADVERSARIAL) \
		--overwrite_output_dir \





