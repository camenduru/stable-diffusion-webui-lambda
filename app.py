# import os
# from subprocess import getoutput

# gpu_info = getoutput('nvidia-smi')
# if("A10G" in gpu_info):
#     os.system(f"pip install -q https://github.com/camenduru/stable-diffusion-webui-colab/releases/download/0.0.15/xformers-0.0.15.dev0+4c06c79.d20221205-cp38-cp38-linux_x86_64.whl")
# elif("T4" in gpu_info):
#     os.system(f"pip install -q https://github.com/camenduru/stable-diffusion-webui-colab/releases/download/0.0.15/xformers-0.0.15.dev0+1515f77.d20221130-cp38-cp38-linux_x86_64.whl")

# os.system(f"sudo git clone -b v2.2 https://github.com/camenduru/stable-diffusion-webui /home/demo/source/stable-diffusion-webui")
# os.chdir(f"/home/demo/source/stable-diffusion-webui")

# os.system(f"wget -q https://github.com/camenduru/webui/raw/main/env_patch.py -O /home/demo/source/env_patch.py")
# os.system(f"sed -i '$a fastapi==0.90.0' /home/demo/source/stable-diffusion-webui/requirements_versions.txt")
# os.system(f"sed -i -e '/import image_from_url_text/r /home/demo/source/env_patch.py' /home/demo/source/stable-diffusion-webui/modules/ui.py")
# os.system(f"sed -i -e '/(modelmerger_interface, \"Checkpoint Merger\", \"modelmerger\"),/d' /home/demo/source/stable-diffusion-webui/modules/ui.py")
# os.system(f"sed -i -e '/(train_interface, \"Train\", \"ti\"),/d' /home/demo/source/stable-diffusion-webui/modules/ui.py")
# os.system(f"sed -i -e '/extensions_interface, \"Extensions\", \"extensions\"/d' /home/demo/source/stable-diffusion-webui/modules/ui.py")
# os.system(f"sed -i -e '/settings_interface, \"Settings\", \"settings\"/d' /home/demo/source/stable-diffusion-webui/modules/ui.py")
# os.system(f'''sed -i -e "s/document.getElementsByTagName('gradio-app')\[0\].shadowRoot/!!document.getElementsByTagName('gradio-app')[0].shadowRoot ? document.getElementsByTagName('gradio-app')[0].shadowRoot : document/g" /home/demo/source/stable-diffusion-webui/script.js''')
# os.system(f"sed -i -e 's/                show_progress=False,/                show_progress=True,/g' /home/demo/source/stable-diffusion-webui/modules/ui.py")
# os.system(f"sed -i -e 's/shared.demo.launch/shared.demo.queue().launch/g' /home/demo/source/stable-diffusion-webui/webui.py")
# os.system(f"sed -i -e 's/ outputs=\[/queue=False, &/g' /home/demo/source/stable-diffusion-webui/modules/ui.py")
# os.system(f"sed -i -e 's/               queue=False,  /                /g' /home/demo/source/stable-diffusion-webui/modules/ui.py")

# # ----------------------------Please duplicate this space and delete this block if you don't want to see the extra header----------------------------
# os.system(f"wget -q https://github.com/camenduru/webui/raw/main/header_patch.py -O /home/demo/source/header_patch.py")
# os.system(f"sed -i -e '/demo:/r /home/demo/source/header_patch.py' /home/demo/source/stable-diffusion-webui/modules/ui.py")
# # ---------------------------------------------------------------------------------------------------------------------------------------------------

# if "IS_SHARED_UI" in os.environ:
#     os.system(f"rm -rfv /home/demo/source/stable-diffusion-webui/scripts/")
    
#     os.system(f"wget -q https://github.com/camenduru/webui/raw/main/shared-config.json -O /home/demo/source/shared-config.json")
#     os.system(f"wget -q https://github.com/camenduru/webui/raw/main/shared-ui-config.json -O /home/demo/source/shared-ui-config.json")

#     os.system(f"wget -q https://huggingface.co/ckpt/anything-v3-vae-swapped/resolve/main/anything-v3-vae-swapped.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/anything-v3-vae-swapped.ckpt")
#     # os.system(f"wget -q {os.getenv('MODEL_LINK')} -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/{os.getenv('MODEL_NAME')}")
#     # os.system(f"wget -q {os.getenv('VAE_LINK')} -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/{os.getenv('VAE_NAME')}")
#     # os.system(f"wget -q {os.getenv('YAML_LINK')} -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/{os.getenv('YAML_NAME')}")
    
#     os.system(f"python launch.py --force-enable-xformers --disable-console-progressbars --enable-console-prompts --ui-config-file /home/demo/source/shared-ui-config.json --ui-settings-file /home/demo/source/shared-config.json --cors-allow-origins huggingface.co,hf.space --no-progressbar-hiding")
# else:

# Please duplicate this space and delete # character in front of the custom script you want to use or add here more custom scripts with same structure os.system(f"wget -q https://CUSTOM_SCRIPT_URL -O /home/demo/source/stable-diffusion-webui/scripts/CUSTOM_SCRIPT_NAME.py")
# os.system(f"wget -q https://gist.github.com/camenduru/9ec5f8141db9902e375967e93250860f/raw/d0bcf01786f20107c329c03f8968584ee67be12a/run_n_times.py -O /home/demo/source/stable-diffusion-webui/scripts/run_n_times.py")

# Please duplicate this space and delete # character in front of the extension you want to use or add here more extensions with same structure os.system(f"git clone https://EXTENSION_GIT_URL /home/demo/source/stable-diffusion-webui/extensions/EXTENSION_NAME")
#os.system(f"git clone https://github.com/camenduru/stable-diffusion-webui-artists-to-study /home/demo/source/stable-diffusion-webui/extensions/stable-diffusion-webui-artists-to-study")
# os.system(f"git clone https://github.com/yfszzx/stable-diffusion-webui-images-browser /home/demo/source/stable-diffusion-webui/extensions/stable-diffusion-webui-images-browser")
# os.system(f"git clone https://github.com/camenduru/deforum-for-automatic1111-webui /home/demo/source/stable-diffusion-webui/extensions/deforum-for-automatic1111-webui")

# Please duplicate this space and delete # character in front of the model you want to use or add here more ckpts with same structure os.system(f"wget -q https://CKPT_URL -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/CKPT_NAME.ckpt")
#os.system(f"wget -q https://huggingface.co/nitrosocke/Arcane-Diffusion/resolve/main/arcane-diffusion-v3.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/arcane-diffusion-v3.ckpt")
#os.system(f"wget -q https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion/resolve/main/Cyberpunk-Anime-Diffusion.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/Cyberpunk-Anime-Diffusion.ckpt")
#os.system(f"wget -q https://huggingface.co/prompthero/midjourney-v4-diffusion/resolve/main/mdjrny-v4.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/mdjrny-v4.ckpt")
#os.system(f"wget -q https://huggingface.co/nitrosocke/mo-di-diffusion/resolve/main/moDi-v1-pruned.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/moDi-v1-pruned.ckpt")
#os.system(f"wget -q https://huggingface.co/Fictiverse/Stable_Diffusion_PaperCut_Model/resolve/main/PaperCut_v1.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/PaperCut_v1.ckpt")
#os.system(f"wget -q https://huggingface.co/lilpotat/sa/resolve/main/samdoesarts_style.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/samdoesarts_style.ckpt")
#os.system(f"wget -q https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float32.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/wd-v1-3-float32.ckpt")
#os.system(f"wget -q https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/sd-v1-4.ckpt")
#os.system(f"wget -q https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.ckpt")
#os.system(f"wget -q https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/sd-v1-5-inpainting.ckpt")

#os.system(f"wget -q https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0-pruned.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/Anything-V3.0-pruned.ckpt")
#os.system(f"wget -q https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/Anything-V3.0.vae.pt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/Anything-V3.0-pruned.vae.pt")

#os.system(f"wget -q https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/768-v-ema.ckpt")
#os.system(f"wget -q https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/768-v-ema.yaml")

# os.system(f"wget -q https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/v2-1_768-ema-pruned.ckpt")
# os.system(f"wget -q https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/v2-1_768-ema-pruned.yaml")

# os.system(f"python launch.py --force-enable-xformers --ui-config-file /home/demo/source/ui-config.json --ui-settings-file /home/demo/source/config.json --disable-console-progressbars --enable-console-prompts --cors-allow-origins huggingface.co,hf.space --no-progressbar-hiding --api --skip-torch-cuda-test")

# os.system(f"sudo python3 launch.py")


import gradio as gr
import os

os.system(f"git clone -b v2.2 https://github.com/camenduru/stable-diffusion-webui /home/demo/source/stable-diffusion-webui")

def greet(name):
    return "Hello " + name + "!"

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")


demo.launch()
