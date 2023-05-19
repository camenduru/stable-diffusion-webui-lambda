import os

os.system(f"git config --global --add safe.directory '*'")
os.system(f"git clone -b v2.2 https://github.com/camenduru/stable-diffusion-webui /home/demo/source/stable-diffusion-webui")
os.chdir(f"/home/demo/source/stable-diffusion-webui")
os.system(f"wget -q https://huggingface.co/ckpt/anything-v4.5-vae-swapped/resolve/main/anything-v4.5-vae-swapped.safetensors -O /home/demo/source/stable-diffusion-webui/models/Stable-diffusion/anything-v4.5-vae-swapped.safetensors")
os.system(f"python launch.py --port 8266 --listen --cors-allow-origins=*")