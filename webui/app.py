from __future__ import annotations

import gradio as gr

from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import pandas as pd
import json


# Load the model and tokenizer
model_path = "/H1/zhouhongli/POEnhancer/models/Qwen2.5-3B-Instruct"
llm = LLM(model=model_path, gpu_memory_utilization=0.5)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)


# Enhanced Theme with more customization, removing unsupported parameters
class EnhancedSeafoam(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.blue,
            secondary_hue=colors.green,
            neutral_hue=colors.gray,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_md,
            text_size=sizes.text_md,
            font="Poppins",
            font_mono="Fira Code",
        )

        # Customizing elements that are supported by the set method
        self.set(
            body_background_fill="repeating-linear-gradient(45deg, *primary_100, *primary_100 10px, *neutral_50 10px, *neutral_50 20px)",
            body_background_fill_dark="repeating-linear-gradient(45deg, *primary_900, *primary_900 10px, *neutral_800 10px, *neutral_800 20px)",
            button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_400, *secondary_400)",
            block_shadow="*shadow_drop_md",
            block_border_width="2px",
        )


css = """
/* General styles */
body {
    background-color: var(--body-background-fill);
    color: var(--neutral-900);
    font-family: 'Poppins', sans-serif;
}

.gr-input, .gr-textarea {
    background-color: var(--neutral-50) !important;
    color: var(--neutral-900) !important;
    border-color: var(--neutral-300) !important;
    border-radius: 8px;
    padding: 10px;
    transition: all 0.3s ease;
}

.gr-input:focus, .gr-textarea:focus {
    border-color: var(--primary-500) !important;
    outline: none;
}

.gr-button {
    background-color: var(--button-primary-background-fill) !important;
    color: var(--neutral-100) !important;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.gr-button:hover {
    background-color: var(--button-primary-background-fill-hover) !important;
}

/* Details section styling */
.details-section {
    margin-top: 20px;
    padding: 20px;
    background-color: var(--neutral-100); /* Light gray background */
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.details-section h3 {
    margin: 0 0 10px 0;
    color: var(--neutral-900);
}

.details-section p {
    margin: 0 0 10px 0;
    color: var(--neutral-600);
    white-space: pre-wrap;
    word-wrap: break-word;
}

.details-section pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    background-color: var(--neutral-200); /* Slightly darker gray background for code blocks */
    padding: 10px;
    border-radius: 8px;
    overflow-x: auto;
    color: var(--neutral-900);
}

/* Markdown styling */
.markdown-body {
    font-size: 16px;
    color: var(--neutral-900);
}

.markdown-body h1 {
    font-size: 2em;
    margin-bottom: 0.5em;
    color: var(--primary-700);
}

.markdown-body h2 {
    font-size: 1.5em;
    margin-bottom: 0.5em;
    color: var(--secondary-700);
}

.markdown-body p {
    margin: 0 0 1em 0;
    color: var(--neutral-600);
}

/* Tab styling */
.tab-container {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.tab-item {
    padding: 10px 20px;
    border: 1px solid var(--neutral-300);
    border-radius: 8px 8px 0 0;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-right: 10px;
}

.tab-item.active {
    background-color: var(--primary-100);
    border-bottom-color: transparent;
    position: relative;
    z-index: 1;
}

.tab-content {
    padding: 20px;
    border: 1px solid var(--neutral-300);
    border-radius: 0 8px 8px 8px;
    background-color: var(--neutral-50);
}
"""

# Conversation Formatting


def create_conversation(instruction, answer1, answer2):
    if not instruction or not answer1 or not answer2:
        raise ValueError(
            "Instruction, Answer 1, and Answer 2 cannot be empty."
        )

    return [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the quality of the answer.",
        },
        {
            "role": "user",
            "content": f"""[Question]
{instruction}
[The Start of Assistant 1's Answer]
{answer1}
[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer2}
[The End of Assistant 2's Answer]

We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.""",
        },
    ]


# Single Evaluation
def evaluate(instruction, answer1, answer2):
    try:
        conversation = create_conversation(instruction, answer1, answer2)
        prompt_token_ids = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        outputs = llm.generate(
            prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
        )
        result = outputs[0].outputs[0].text.strip()

        scores = result.splitlines()[0].split()

        if len(scores) == 2:
            score1, score2 = map(float, scores)
            verdict = (
                "Assistant 1 is better 🎉"
                if score1 > score2
                else (
                    "Assistant 2 is better 🎉"
                    if score2 > score1
                    else "Both assistants are equally good! 🤝"
                )
            )
        else:
            verdict = "Error parsing scores."

        # Create the full prompt as it was passed to the model for evaluation.
        full_prompt = "\n".join([msg["content"] for msg in conversation])

        details = """<div class="details-section">
    <h3>👨🏽‍💻 User</h3>
    <p>{}</p>
    <h3>🧑‍⚖️ Judge Model</h3>
    <p>{}</p>
</div>
""".format(
            full_prompt.replace('>', '&gt;').replace(
                '<', '&lt;').replace('\n', '<br>'),
            result.replace('\n', '<br>')
        )

        return verdict, details
    except Exception as e:
        # Return error message and details
        return (
            "Error during evaluation",
            f"<pre>Details: {str(e)}</pre>",
        )


# Batch Evaluation
def evaluate_batch(file, output_path):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file.name)
    elif file.name.endswith(".json"):
        with open(file.name, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        return "仅支持 CSV 或 JSON 格式的文件"

    results = []
    for _, row in df.iterrows():
        instruction = row.get("instruction", "")
        answer1 = row.get("answer1", "")
        answer2 = row.get("answer2", "")

        if not instruction or not answer1 or not answer2:
            results.append("Invalid row: Missing data")
            continue

        try:
            conversation = create_conversation(instruction, answer1, answer2)
            prompt_token_ids = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            outputs = llm.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
            result = outputs[0].outputs[0].text.strip()
        except Exception as e:
            result = f"Error: {str(e)}"

        results.append(result)

    output_df = pd.DataFrame(
        {
            "Instruction": df.get("instruction", []),
            "Answer 1": df.get("answer1", []),
            "Answer 2": df.get("answer2", []),
            "Evaluation Result": results,
        }
    )
    try:
        output_df.to_csv(output_path, index=False)
        return f"评估结果已保存到 {output_path}"
    except Exception as e:
        return f"保存文件时出错：{str(e)}"


# Gradio Interface
details_visible = False  # Track visibility state of the details section


def toggle_details():
    global details_visible
    details_visible = not details_visible
    return (
        # Update visibility of the details section
        gr.update(visible=details_visible),
        "Hide Details" if details_visible else "Show Details",  # Update button text
    )


with gr.Blocks(theme=EnhancedSeafoam(), css=css) as demo:
    gr.Markdown("# LLM Evaluation Web Application")
    gr.Markdown(
        "### A web application for evaluating the outputs of large language models."
    )

    with gr.Tab("Manual Input"):
        instruction_input = gr.Textbox(
            label="Instruction",
            placeholder="Enter the instruction here...",
            lines=3,
        )
        answer1_input = gr.Textbox(
            label="Assistant 1",
            placeholder="Enter the first answer here...",
            lines=3,
        )
        answer2_input = gr.Textbox(
            label="Assistant 2",
            placeholder="Enter the second answer here...",
            lines=3,
        )
        result_output = gr.Textbox(
            label="Evaluation Summary", interactive=False)
        details_output = gr.HTML(
            "<div class='details-section'><h3>Details will appear here</h3></div>",
            visible=False,  # Default to hidden
            elem_classes=["details-section"],
        )

        evaluate_btn = gr.Button("Evaluate")
        evaluate_btn.click(
            evaluate,
            inputs=[instruction_input, answer1_input, answer2_input],
            outputs=[result_output, details_output],
        )

        # Add a button to toggle details visibility
        details_button = gr.Button("Show Details")
        details_button.click(
            toggle_details,
            outputs=[details_output, details_button],
        )

    with gr.Tab("Batch Evaluation"):
        file_input = gr.File(label="Upload CSV or JSON File")
        save_path_input = gr.Textbox(
            label="Save Path", placeholder="Enter output file path"
        )
        batch_result_output = gr.Textbox(
            label="Batch Result", interactive=False)
        batch_evaluate_btn = gr.Button("Start Batch Evaluation")
        batch_evaluate_btn.click(
            evaluate_batch,
            inputs=[file_input, save_path_input],
            outputs=batch_result_output,
        )

    gr.Markdown(
        "<div style='font-size: 18px; text-align: center;'>Designed by Hongli Zhou</div>"
    )

if __name__ == "__main__":
    demo.launch()
