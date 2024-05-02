import torch
import json
import sys
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
import argparse
from tqdm import tqdm
import json, os
from datasets import load_dataset
from torch.utils.data import DataLoader
import copy
import pdb
import logging
import re

from peft import PeftModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model, Accelerator

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--log_file', type=str, required=True)
parser.add_argument('--result_file', type=str, required=True)
parser.add_argument('--llama', action="store_true")
parser.add_argument('--dev_data_path', type=str, required=True)
args = parser.parse_args()

max_new_tokens = 200
generation_config = dict(
    temperature=0.001,
    top_k=30,
    top_p=0.85,
    do_sample=True,
    repetition_penalty=1.1,
    max_new_tokens=max_new_tokens
)


def fix_sql(sql):
    sql = "SELECT " + sql
    sql = sql.replace("\n", " ")
    sql = sql.replace("  ", " ")
    return sql


def extract_string(sentence):
    index = sentence.find(";")  # 查找第一个分号的索引位置
    if index != -1:
        substring = sentence[:index]  # 提取分号之前的子字符串
        return substring
    else:
        return sentence  # 如果没有找到分号，则返回 None


def get_questions(val_data):
    questions = []
    answers = []
    db_ides = []
    for data in val_data:
        db_ides.append(data['db_id'])
        input = data["instruction"]
        sentence_ids = tokenizer.encode(input, add_special_tokens=False)
        questions.append(sentence_ids)
        output = data["output"]
        answers.append(output)

    return db_ides, questions, answers


def print_rank_0(msg, log_file, rank=0):
    if rank <= 0:
        with open(log_file, 'a') as f:
            print(msg)
            f.write(msg + '\n')
            f.close()


def write_cov(sql_dict, file):
    with open(file, encoding="utf-8", mode='a') as f:
        f.write(json.dumps(sql_dict, ensure_ascii=False) + "\n")
        f.close()


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')


    # cuda_list = '0,1'.split(',')
    # memory = '20GiB'
    # max_memory = {int(cuda): memory for cuda in cuda_list}
    # model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # no_split_module_classes = LlamaForCausalLM._no_split_modules

    log_file = args.log_file
    result_file = args.result_file
    accelerator = Accelerator()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Start inference , loading the model")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type,
    #                                                   trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type,
                                           trust_remote_code=True,device_map='auto')

    model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype=load_type)

    # with init_empty_weights():
    #     base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type,
    #                                                       trust_remote_code=True)
    #
    #     model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype=load_type)  # 加载到meta设备中，不需要耗时，不需要消耗内存和显存
    #
    # device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
    # load_checkpoint_in_model(model, args.model_name_or_path, device_map=device_map)
    # model = dispatch_model(model, device_map=device_map)


    # if device == torch.device('cpu'):
    #     model.float()
    # model.to(device)
    model.eval()
    logger.info("Load model successfully")

    space_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    space_tensor = torch.LongTensor([[space_id]]).to(device)

    # val_data = load_dataset("json", data_files=args.dev_data_path, cache_dir='../')
    val_data = json.load(open(args.dev_data_path, "r"))
    db_ids, questions, answers = get_questions(val_data)
    test_dict = {}
    for i, inputs in enumerate(questions):
        model_ans = None
        print_rank_0(
            "=============================== question: {}=====================================================================".format(
                i), log_file)
        inputs = torch.LongTensor(inputs).unsqueeze(0).to(device)
        print_rank_0(
            "===============================model input:=====================================================================",
            log_file)
        print_rank_0(tokenizer.decode(inputs[0], skip_special_tokens=True), log_file)
        generation_output = model.generate(input_ids=inputs, **generation_config)[0]
        generate_text = tokenizer.decode(generation_output, skip_special_tokens=True)
        input_len = inputs.size()[1]
        gen_len = len(generation_output) - input_len
        model_ans = generation_output[-gen_len:]
        ans_text = tokenizer.decode(model_ans, skip_special_tokens=True)
        ans_text = fix_sql(ans_text)
        ans_text = extract_string(ans_text)
        if not ans_text:
            ans_text = "null"
        print_rank_0(
            "===============================sql output:=====================================================================",
            log_file)
        print_rank_0(ans_text, log_file)
        infer_dict = {"cov_id": i, "db_id": db_ids[i], "sql_output": ans_text, "golden_sql": answers[i]}
        test_dict[str(i)] = ans_text + ";\t----- bird -----\t" + db_ids[i]

        print_rank_0(
            "=============================== GOLDEN SQL:=====================================================================",
            log_file)
        print_rank_0(answers[i] + "\n", log_file)
        write_cov(infer_dict, result_file)

    logger.info("End inference")

