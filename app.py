#AI STORYTELLER – Gradio + HuggingFace 
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model 
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=10 if torch.cuda.is_available() else -1
)

# Story generation function

def generate_story(prompt, max_length, temperature, top_p, seed):
    if not prompt.strip():
        return " Please enter a prompt."

    if seed != 0:
        torch.manual_seed(seed)

    output = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
    )[0]["generated_text"]

    # Remove repeated prompt
    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    return output

# Gradio UI

with gr.Blocks(title="AI Storyteller") as demo:
    gr.Markdown("""✨ AI Storyteller  
    Generate short stories using a simple Generative AI model!""")

    prompt = gr.Textbox(
        label="Enter story prompt",
        placeholder="Example: A young explorer discovers a hidden library beneath the city...",
        lines=4
    )

    with gr.Row():
        max_length = gr.Slider(50, 600, value=200, step=25, label="Max Length")
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        seed = gr.Number(value=0, label="Random Seed (0 = random)")

    btn = gr.Button("Generate Story ✨")

    output = gr.Textbox(label="Generated Story", lines=12)

    btn.click(
        generate_story,
        inputs=[prompt, max_length, temperature, top_p, seed],
        outputs=output
    )
# Launch
demo.launch(share=True)

