import gradio as gr
from vllm import LLM, SamplingParams

# 加载 vllm 模型
llm = LLM(model="/H1/zhouhongli/PORM/output/Llama-2-7b-chat-hf-helpsteer_et",
          gpu_memory_utilization=0.5)


def evaluate(instruction, answer1, answer2):
    # 创建提示
    prompt = """<s> [INST] <<SYS>>
You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.
<</SYS>>

Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.

Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.

# Instruction:
{instruction}

# Output (a):
{answer1}

# Output (b):
{answer2}

# Which is better, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)": [/INST]"""

    # 调用 vllm 生成
    sampling_params = SamplingParams(max_tokens=1024)
    outputs = llm.generate(prompt, sampling_params)

    # 获取生成的文本
    result = outputs[0].outputs[0].text

    # 提取评估结果
    if "Output (a)" in result:
        return "回答 1 更好"
    elif "Output (b)" in result:
        return "回答 2 更好"
    else:
        return "无法确定哪个回答更好"


# 创建 Gradio 页面
demo = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.Textbox(label="指令"),
        gr.Textbox(label="回答 1"),
        gr.Textbox(label="回答 2")
    ],
    outputs=gr.Textbox(label="评估结果"),
    title="LLM 评估系统",
    description="请输入指令和两条回答，系统将使用 VLLM 进行评估"
)

if __name__ == "__main__":
    demo.launch()
