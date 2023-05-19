# import os
# os.system(f"git config --global --add safe.directory '*'")
# os.system(f"git clone -b v2.2 https://github.com/camenduru/stable-diffusion-webui /home/demo/source/stable-diffusion-webui")
# os.chdir(f"/home/demo/source/stable-diffusion-webui")
# os.system(f"wget -q https://huggingface.co/ckpt/anything-v4.5-vae-swapped/resolve/main/anything-v4.5-vae-swapped.safetensors -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/anything-v4.5-vae-swapped.safetensors")
# os.system(f"python launch.py --port 8266 --listen --cors-allow-origins=*")

import gradio as gr
from subprocess import getoutput

def greet(name):
    return "Hello " + name + "!"

def run(command):
    out = getoutput(f"{command}")
    return out

with gr.Blocks() as demo:
    command = gr.Textbox(show_label=False, max_lines=1, placeholder="command")
    out_text = gr.Textbox(show_label=False)
    btn_run = gr.Button("run command")
    btn_run.click(run, inputs=command, outputs=out_text)

demo.launch()