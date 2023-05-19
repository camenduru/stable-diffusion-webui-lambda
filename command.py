import gradio as gr
from subprocess import getoutput

def run(command):
    out = getoutput(f"{command}")
    return out

with gr.Blocks() as demo:
    command = gr.Textbox(show_label=False, max_lines=1, placeholder="command")
    out_text = gr.Textbox(show_label=False)
    btn_run = gr.Button("run command")
    btn_run.click(run, inputs=command, outputs=out_text)

demo.launch()