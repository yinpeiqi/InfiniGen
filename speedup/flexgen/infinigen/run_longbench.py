import os
import torch
import argparse
from tqdm import tqdm
import json
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from flexgen.compression import CompressionConfig
from flexgen.llama_config import get_llama_config
from flexgen.pytorch_backend import LlamaTorchDevice, TorchDisk, TorchMixedDevice
from flexgen.flex_opt import Policy, get_filename, get_inputs
from flexgen.flex_llama import LlamaLM
from flexgen.timer import timers
from flexgen.utils import ExecutionEnv, GB, str2bool, project_decode_latency, write_benchmark_log

all_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]


def run_flexgen(args):
    model_index = 0
    dataset_name = args.dataset_name
    select_method = "infinigen"
    out_path = "results/longbench_result_{1}_{2}_{3}".format(
        select_method, model_index, dataset_name, args.alpha)

    print(f"<run_infinigen>: args.model: {args.model}, dataset: {args.dataset_name}, alpha: {args.alpha}")
    num_prompts = args.num_gpu_batches * args.gpu_batch_size

    gpu = LlamaTorchDevice("cuda:0")
    cpu = LlamaTorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    llama_config = get_llama_config(args.model, hf_token=args.hf_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token, padding_side="left")
    tokenizer.pad_token_id = llama_config.pad_token_id
    tokenizer.eos_token_id = llama_config.eos_token_id
    model = LlamaLM(llama_config, env, args.path, policy, args.partial_weight_ratio, args.alpha, args.max_num_kv)

    # Task and policy
    warmup_inputs = get_inputs(2048, num_prompts, tokenizer, args.warmup_input_path)

    data = load_dataset('THUDM/LongBench', dataset_name, split='test')
    dataset2prompt = json.load(open("longbench2prompt.json", "r"))
    dataset2maxlen = json.load(open("longbench2maxlen.json", "r"))
    prompt_format = dataset2prompt[dataset_name]
    max_output_len = dataset2maxlen[dataset_name]
    preds = []
    batch_size = 1
    batch_prompts = []
    batch_json = []

    print("warmup - generate")
    model.generate(warmup_inputs, max_new_tokens=1, warmup=True)
    tot_spar = []
    for json_obj in tqdm(data):
        if json_obj['language'] == "en":
            batch_prompts.append(prompt_format.format(**json_obj))
            batch_json.append(json_obj)
            if len(batch_prompts) == batch_size:
                torch.cuda.empty_cache()
                inputs_ids = tokenizer(batch_prompts[0]).input_ids
                if (len(inputs_ids) < 32000):
                    if (len(inputs_ids) < 13400):
                        output_ids = model.generate((inputs_ids,), max_new_tokens=max_output_len)
                        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                        sparsity_rate = model.get_sparsity_and_clear()
                        assert (sparsity_rate != 0)
                        print(sparsity_rate)
                        tot_spar.append(sparsity_rate)
                        print(outputs[0].replace("\n", "\\n "))
                        for idx, output in enumerate(outputs):
                            num_prefill_tokens = len(inputs_ids)
                            num_decode_tokens = len(output_ids[0])
                            preds.append({"pred": outputs[0], 
                                        "answers": batch_json[idx]["answers"], 
                                        "all_classes": batch_json[idx]["all_classes"], 
                                        "length": batch_json[idx]["length"], 
                                        "token_length": num_prefill_tokens,
                                        "decode_length": num_decode_tokens,
                                        "actual_length": num_prefill_tokens + num_decode_tokens,
                                        "sparisty_rate": sparsity_rate})
                else:
                    print("too long! continue")
                    preds.append({"pred": "OOM", 
                                "answers": batch_json[idx]["answers"], 
                                "all_classes": batch_json[idx]["all_classes"], 
                                "length": batch_json[idx]["length"], 
                                "token_length": 0,
                                "decode_length": 0,
                                "actual_length": 0,
                                "sparisty_rate": 0})
                batch_prompts.clear()
                batch_json.clear()
    with open(out_path, "w+", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')
    print("tot sparsity:", sum(tot_spar) / len(tot_spar))
    env.close_copy_threads()
    return

def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
        help="The model name.")
    parser.add_argument("--hf-token", type=str,
        help="The huggingface token for accessing gated repo.")
    parser.add_argument("--path", type=str, default="~/llama_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")
    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--warmup-input-path", type=str)

    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--partial-weight-ratio", type=float, default=0.3)
    parser.add_argument("--max-num-kv", type=int, default=400)
    parser.add_argument("--dataset-name", type=str)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    with torch.no_grad():
        run_flexgen(args)