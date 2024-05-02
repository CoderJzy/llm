CUDA_VISIBLE_DEVICES=0 python run_model.py \
	--model_name_or_path /root/autodl-tmp/model/WizardCoder-15B-V1 \
	--ckpt_path /root/output/lora_2 \
	--log_file /root/log/lora.txt \
	--result_file /root/result/lora.txt \
	--dev_data_path /root/data/test.json \
