from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('AI-ModelScope/WizardCoder-15B-V1.0', cache_dir='/root/autodl-tmp/model')
