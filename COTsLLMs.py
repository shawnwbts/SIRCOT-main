from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import random
import time
import math

random.seed(1234)

client = OpenAI(
    api_key='your_api_key',
    base_url = "your_url"
)

def generate_summaries_scence(testcode,testsummary):
    sysContent = (
        "You are an expert in smart contract analysis. "
        "Classify the given code and its summarization into one of five categories. "
        "Categories:\n"
        "1. Functional – describes primary purpose (what it does).\n"
        "2. Interface and Usability – explains usage, inputs/outputs, access (how to use it).\n"
        "3. Safety and Constraint – highlights risks, checks, or safeguards (what to watch out for).\n"
        "4. Execution and Interaction – shows system-level behavior, contract/external interactions.\n"
        "5. Maintenance – notes design rationale, debugging, upgrade, or future evolution."
    )
    userContent = (
            "Code:\n" + testcode +
            "\nSummary:\n" + testsummary +
            "\nOutput only one category name: Functional, Interface and Usability, "
            "Safety and Constraint, Execution and Interaction, or Maintenance."
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': sysContent},
                {'role': 'user', 'content': userContent},
            ],
            model='gpt-3.5-turbo',
            temperature=0.0  # 分类任务建议用0温度
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None


def generate_dynamic_cot(testcode,testsummary):
    # 任务类型对应的推理步骤模板
    task_templates = {
        "Functional":
            "Step 1: Identify the core functionality of the function. Step 2: Determine whether it implements business logic. Step 3: Summarize the purpose in one concise sentence.",
        "Interface and Usability":
            "Step 1: Check if the function defines access restrictions. Step 2: Identify invocation attributes (payable, view, external). Step 3: Explain input/output meaning and constraints. Step 4: Provide a short, user-oriented annotation.",
        "Safety and Constraint":
            "Step 1: Detect constraints on values, gas, or execution order. Step 2: Check for require/revert and document exceptions. Step 3: Note if the function is deprecated. Step 4: Highlight any security patterns.",
        "Execution and Interaction":
            "Step 1: Identify whether the function is fallback, receive, or cross-contract interaction. Step 2: Explain its role in external calls, ETH receiving, or delegatecall/oracle callbacks. Step 3: Clarify execution flow to avoid ambiguity.",
        "Maintenance":
            "Step 1: Look for metadata such as versioning, authorship, SPDX identifiers. Step 2: Explain their relevance for contract maintenance and long-term usability. Step 3: Provide annotation focusing on documentation rather than execution."
    }

    task_type = generate_summaries_scence(testcode, testsummary)
    print(task_type)
    # 根据任务类型选择COT模板
    cot = task_templates.get(task_type)
    if cot is None:
        print(f"Error: No matching task type found for {task_type}")
        cot = "None"  # 你可以根据需求设置一个默认值

        # 返回动态生成的COT
    return cot

def generate_summaries_chain_of_thought(example_source1, example_target1, example_source2, example_target2,
                                         example_source3, example_target3, example_source4, example_target4, code,
                                         example_choose):
    sysContent = (
        f"Main Task: Summarize the smart contract code in ONE concise sentence. {code}\n"
       f"Primary rule: The comment must not exceed {example_choose} characters.\n"
        "Secondary rule: The comment must be exactly one sentence, short and clear.\n"
        "Guidance: Use the given examples and scenario guidance as inspiration, but do not explain..\n"
        "Output Requirement: Return only the comment, without additional text."
    )
    fewshot_prompt = (
        f"#example code 1:\n{example_source1}\n#example summarization 1:\n{example_target1}\n"
        f"#example code 2:\n{example_source2}\n#example summarization 2:\n{example_target2}\n"
        f"#example code 3:\n{example_source3}\n#example summarization 3:\n{example_target3}\n"
        f"#example code 4:\n{example_source4}\n#example summarization 4:\n{example_target4}\n"
    )


    cot_comment = generate_dynamic_cot(code, example_choose)
    scenario_prompt = f"Scenario guidance: {cot_comment}"

    final_prompt = (
        f"{scenario_prompt}\n\n"
        f"Now generate one concise comment (<= {example_choose} characters)."
    )


    try:
        # 调用 OpenAI API
        response = client.chat.completions.create(
            messages=[{'role': 'system', 'content': sysContent},
                      {'role': 'user', 'content': fewshot_prompt},
                      {'role': 'user', 'content': final_prompt}],
            model='gpt-3.5-turbo',
            temperature=0.2,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    df = pd.read_csv('data/example_all.csv')
    example_code1 = df['code1'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment1 = df['comment1'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code2 = df['code2'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment2 = df['comment2'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code3 = df['code3'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment3 = df['comment3'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code4 = df['code4'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment4 = df['comment4'].tolist()

    df = pd.read_csv('data/test_data_function.csv', header=None,encoding='ISO-8859-1')
    source_codes = df[0].tolist()

    df = pd.read_csv('data/test_data_comment.csv', header=None,encoding='ISO-8859-1')
    example = df[0].tolist()

    batch_size = 50

    num_batches = math.ceil(len(source_codes) / batch_size)
    print(num_batches)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(source_codes))

        source_batch = source_codes[start_index:end_index]
        example_batch = example[start_index:end_index]

        python_codes = []
        for i in tqdm(range(len(source_batch)), mininterval=0.1, maxinterval=1):
            python_codes.append(generate_summaries_chain_of_thought(example_code1[i], example_comment1[i], example_code2[i],example_comment2[i],example_code3[i],example_comment3[i],example_code4[i],example_comment4[i],source_batch[i], example_batch[i]))
        time.sleep(1)
        # print(python_codes)
        df = pd.DataFrame(python_codes)
        if batch_index == 0:
            with open('result/sml1.csv', 'w', newline='\n') as f:
                df.to_csv(f, index=False, header=True)
        else:
            with open('result/sml1.csv', 'a', newline='\n') as f:
                df.to_csv(f, index=False, header=True)