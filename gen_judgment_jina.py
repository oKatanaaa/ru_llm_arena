import json
import yaml
import argparse
import os
import re
from tqdm import tqdm
from transformers import AutoModel
import torch

from tqdm import tqdm

from utils import (
    load_questions,
    load_questions,
    load_model_answers,
    make_config,
)

prompt_template = """
<user prompt>
{question_1}
<end>
<assistant A answer>
{answer_1}
<end>
<assistant B answer>
{answer_2}
<end>
""".strip()


def get_score(model, jina_prompt):
    with torch.no_grad():
        judgment = model([jina_prompt])[0].argmax().item()

    return {
        0: "A<B",
        1: "A=B",
        2: "B>A"
    }[judgment]


def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    output_file = args["output_file"]
    model = args["judge_model"]
    # model = configs["judge_model"]
    #print('judge', model)

    num_games = 2 if configs["pairwise"] else 1

    output = {
        "question_id":question["question_id"],
        "model":answer["model_id"],
        "judge": "jina",
        "games":[]
        }

    for game in range(num_games):

        prompt_args = {}

        for i, turn in enumerate(question["turns"]):
            prompt_args[f"question_{i+1}"] = turn["content"]
        base = 1

        if baseline:
            if game % 2 == 1: # swap position
                temp = baseline
                baseline = answer
                answer = temp

            for i, turn in enumerate(baseline["choices"][0]["turns"]):
                prompt_args[f"answer_{i+1}"] = turn["content"]
                base += 1
        if answer:
            for i, turn in enumerate(answer["choices"][0]["turns"]):
                prompt_args[f"answer_{i+base}"] = turn["content"]

        jina_prompt = prompt_template.format(**prompt_args)

        score = get_score(
            model,
            jina_prompt
        )

        result = {
            "user_prompt": jina_prompt,
            "judgment": "none",
            "score":score
        }
        output["games"].append(result)

    with open(output_file, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/judge_config.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
          + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}, '
          + f"jina_device: {configs['jina_device']}, jina_model: {configs['jina_model']}")

    if configs["regex_pattern"]:
        pattern = re.compile(configs["regex_pattern"])

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")
    ref_answer_dir = os.path.join("data", configs["bench_name"], "reference_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
        
    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]
    
    output_files = {}
    output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    jina_judge = AutoModel.from_pretrained(configs["jina_model"], trust_remote_code=True)
    jina_judge.to(configs["jina_device"])
    jina_judge.eval()

    for model in models:
        count = 0
        for question in tqdm(questions, desc=f"Evaluating {model}"):
            question_id = question["question_id"]

            kwargs = {}
            kwargs["question"] = question
            if model in model_answers and not question_id in model_answers[model]:
                print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                continue

            if model in existing_judgments and question_id in existing_judgments[model]:
                count += 1
                continue

            kwargs["answer"] = model_answers[model][question_id]
            if ref_answers:
                kwargs["reference"] = [ref_answer[question_id] for ref_answer in ref_answers]
                assert len(kwargs["reference"]) == len(configs["ref_model"])
            else:
                kwargs["reference"] = None
            if configs["baseline"]:
                kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][question_id]
            else:
                kwargs["baseline_answer"] = None
            kwargs["configs"] = configs
            kwargs["output_file"] = output_files[model]
            kwargs["regex_pattern"] = pattern
            kwargs["judge_model"] = jina_judge
            judgment(**kwargs)

        if count > 0:
            print(f"{count} number of existing judgments")