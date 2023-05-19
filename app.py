import os, gdown, gc
import numpy as np
import gradio as gr
from diffusers import FlaxStableDiffusionPipeline, StableDiffusionPipeline
import torch
from safetensors.torch import save_file, load_file
from huggingface_hub import model_info, create_repo, create_branch, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

def download_ckpt(ckpt_url):
    if "drive.google.com" in ckpt_url:
        gdown.download(url=ckpt_url, output="model.ckpt", quiet=False, fuzzy=True)
    else:
        os.system(f"wget {ckpt_url} -O model.ckpt")
    return "download ckpt done!"

def download_vae(vae_url):
    if "drive.google.com" in vae_url:
        gdown.download(url=vae_url, output="vae.ckpt", quiet=False, fuzzy=True)
    else:
        os.system(f"wget {vae_url} -O vae.ckpt")
    return "download vae done!"

def to_pt():
    os.system("wget -q https://raw.githubusercontent.com/huggingface/diffusers/v0.13.1/scripts/convert_original_stable_diffusion_to_diffusers.py")
    os.system(f"python3 convert_original_stable_diffusion_to_diffusers.py --checkpoint_path model.ckpt --dump_path pt")
    return "convert to pt done!"

def from_safetensors_to_pt():
    os.system("wget -q https://raw.githubusercontent.com/huggingface/diffusers/v0.13.1/scripts/convert_original_stable_diffusion_to_diffusers.py")
    os.system(f"python3 convert_original_stable_diffusion_to_diffusers.py --from_safetensors --checkpoint_path model.safetensors --dump_path pt")
    return "convert to pt done!"

def from_ckpt_to_safetensors():
    os.system("wget -q https://raw.githubusercontent.com/huggingface/diffusers/v0.13.1/scripts/convert_original_stable_diffusion_to_diffusers.py")
    os.system(f"python3 convert_original_stable_diffusion_to_diffusers.py --checkpoint_path model.ckpt --to_safetensors --dump_path safetensors")
    return "convert to safetensors done!"

def from_safetensors_to_safetensors():
    os.system("wget -q https://raw.githubusercontent.com/huggingface/diffusers/v0.13.1/scripts/convert_original_stable_diffusion_to_diffusers.py")
    os.system(f"python3 convert_original_stable_diffusion_to_diffusers.py --from_safetensors --checkpoint_path model.safetensors --to_safetensors --dump_path safetensors")
    return "convert to safetensors done!"

def from_safetensors_to_emaonly(safetensors_emaonly_name):
    os.system("mkdir safetensors")
    tensors = load_file("model.safetensors")
    filtered_only_ema = {k: v for k, v in tensors.items() if not k.startswith("model.")}
    save_file(filtered_only_ema, f"safetensors/{safetensors_emaonly_name}-emaonly.safetensors")
    return "convert to safetensors emaonly done!"

def swap_ckpt_vae(ckpt_name):
    os.system("mkdir ckpt")
    model = torch.load("model.ckpt", map_location="cpu")
    if "state_dict" in model:
      sd = model["state_dict"]
    else:
      sd = model
    full_model = False
    vae_model = torch.load("vae.ckpt", map_location="cpu")
    vae_sd = vae_model['state_dict']
    for vae_key in vae_sd:
      if vae_key.startswith("first_stage_model."):
        full_model = True
        break
    for vae_key in vae_sd:
      sd_key = vae_key
      if full_model:
        if not sd_key.startswith("first_stage_model."):
          continue
      else:
        if sd_key not in sd:
          sd_key = "first_stage_model." + sd_key
      if sd_key not in sd:
        continue
      sd[sd_key] = vae_sd[vae_key]
    torch.save(model, f"ckpt/{ckpt_name}-vae-swapped.ckpt")
    del model
    del vae_model
    del sd
    del vae_sd
    gc.collect()
    return "swap ckpt vae done!"

def push_pt(model_to, token, branch):
    try:
        repo_exists = True
        r_info = model_info(model_to, token=token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
        else:
            create_repo(model_to, private=True, token=token)
    try:
        branch_exists = True
        b_info = model_info(model_to, revision=branch, token=token)
    except RevisionNotFoundError:
        branch_exists = False
    finally:
        if branch_exists:
            print(b_info)
        else:
            create_branch(model_to, branch=branch, token=token)
    upload_folder(folder_path="pt", path_in_repo="", revision=branch, repo_id=model_to, commit_message=f"pt - camenduru/converter", token=token)
    return "push pt done!"
    
def delete_pt():
    os.system(f"rm -rf pt")
    return "delete pt done!"
    
def clone_pt(model_url):
    os.system("git lfs install")
    os.system(f"git clone https://huggingface.co/{model_url} pt")
    return "clone pt done!"
    
def pt_to_flax():
    pipe, params = FlaxStableDiffusionPipeline.from_pretrained("pt", from_pt=True)
    pipe.save_pretrained("flax", params=params)
    return "convert to flax done!"

def push_flax(model_to, token, branch):
    try:
        repo_exists = True
        r_info = model_info(model_to, token=token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
        else:
            create_repo(model_to, private=True, token=token)
    try:
        branch_exists = True
        b_info = model_info(model_to, revision=branch, token=token)
    except RevisionNotFoundError:
        branch_exists = False
    finally:
        if branch_exists:
            print(b_info)
        else:
            create_branch(model_to, branch=branch, token=token)
    upload_folder(folder_path="flax", path_in_repo="", revision=branch, repo_id=model_to, commit_message=f"flax - camenduru/converter", token=token)
    return "push flax done!"

def delete_flax():
    os.system(f"rm -rf flax")
    return "delete flax done!"
    
def flax_to_pt():
    pipe = StableDiffusionPipeline.from_pretrained("flax", from_flax=True, safety_checker=None)
    pipe.save_pretrained("pt")
    return "convert to pt done!"
    
def clone_flax(model_url):
    os.system("git lfs install")
    os.system(f"git clone https://huggingface.co/{model_url} flax")
    return "clone flax done!"
    
def to_ckpt(ckpt_name):
    os.system("wget -q https://raw.githubusercontent.com/huggingface/diffusers/v0.13.1/scripts/convert_diffusers_to_original_stable_diffusion.py")
    os.system("mkdir ckpt")
    os.system(f"python3 convert_diffusers_to_original_stable_diffusion.py --model_path pt --checkpoint_path ckpt/{ckpt_name}.ckpt")
    return "convert to ckpt done!"

def push_ckpt(model_to, token, branch):
    try:
        repo_exists = True
        r_info = model_info(model_to, token=token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
        else:
            create_repo(model_to, private=True, token=token)
    try:
        branch_exists = True
        b_info = model_info(model_to, revision=branch, token=token)
    except RevisionNotFoundError:
        branch_exists = False
    finally:
        if branch_exists:
            print(b_info)
        else:
            create_branch(model_to, branch=branch, token=token)    
    upload_folder(folder_path="ckpt", path_in_repo="", revision=branch, repo_id=model_to, commit_message=f"ckpt - camenduru/converter", token=token)
    return "push ckpt done!"
    
def delete_ckpt():
    os.system(f"rm -rf ckpt")
    return "delete ckpt done!"

def to_safetensors(safetensors_name):
    os.system("mkdir safetensors")
    weights = torch.load("model.ckpt", map_location="cpu")
    if "state_dict" in weights:
        weights = weights["state_dict"]
    save_file(weights, f"safetensors/{safetensors_name}.safetensors")
    return "convert to safetensors done!"

def push_safetensors(model_to, token, branch):
    try:
        repo_exists = True
        r_info = model_info(model_to, token=token)
    except RepositoryNotFoundError:
        repo_exists = False
    finally:
        if repo_exists:
            print(r_info)
        else:
            create_repo(model_to, private=True, token=token)
    try:
        branch_exists = True
        b_info = model_info(model_to, revision=branch, token=token)
    except RevisionNotFoundError:
        branch_exists = False
    finally:
        if branch_exists:
            print(b_info)
        else:
            create_branch(model_to, branch=branch, token=token)
    upload_folder(folder_path="safetensors", path_in_repo="", revision=branch, repo_id=model_to, commit_message=f"safetensors - camenduru/converter", token=token)
    return "push safetensors done!"

def delete_safetensors():
    os.system(f"rm -rf safetensors")
    return "delete safetensors done!"

def download_safetensors(safetensors_url):
    if "drive.google.com" in safetensors_url:
        gdown.download(url=ckpt_url, output="model.safetensors", quiet=False, fuzzy=True)
    else:
        os.system(f"wget {safetensors_url} -O model.safetensors")
    return "download safetensors done!"

def from_safetensors_to_ckpt(ckpt_name):
    weights = load_file("model.safetensors", device="cpu")
    os.system("mkdir ckpt")
    torch.save(weights, f"ckpt/{ckpt_name}.ckpt")
    return "convert to ckpt done!"

def delete_all():
    delete_pt()
    delete_flax()
    delete_ckpt()
    delete_safetensors()
    return "delete all done!"
    
block = gr.Blocks()

with block:
    gr.Markdown(
    """
    ## 🚨 Please first click delete all button 🚨 Thanks to 🤗 ❤ Now with CPU Upgrade! 🎉 <a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/camenduru/converter?duplicate=true"><img style="margin: 0" src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>&nbsp;&nbsp;<a style="display:inline-block" href="https://github.com/camenduru/converter-colab" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="margin-bottom: 0px; margin-top: 0px;"></a>&nbsp;&nbsp;<a style="display:inline-block" href="https://patreon.com/camenduru"><img style="margin: 0" alt="Become A Patreon" src="https://badgen.net/badge/become/a%20patron/F96854"></a>&nbsp;&nbsp;<a style="display:inline-block" href="https://ko-fi.com/camenduru" target="_blank"><img style="margin: 0" alt="Buy a Coffee" src="https://badgen.net/badge/buy/a%20coffee/green?icon=kofi"></a>
    🐣 Please follow me for new updates <a href="https://twitter.com/camenduru">https://twitter.com/camenduru</a>
    """)
    with gr.Row().style(equal_height=True):
        btn_delete_all = gr.Button("Delete ALL")
        out_all = gr.Textbox(show_label=False)
        btn_delete_all.click(delete_all, outputs=out_all)
    gr.Markdown(
    """
    ### ckpt to diffusers pytorch
    ckpt_url = <small>https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.ckpt or https://drive.google.com/file/d/file-id/view?usp=share_link or "https://civitai.com/api/download/models/5616?type=Model&format=PickleTensor"</small><br />
    pt_model_to = camenduru/openjourney <br />
    branch = main <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_ckpt_url = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_url")
                text_pt_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="pt_model_to")
                text_pt_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="pt_branch")
                text_pt_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_pt = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_download_ckpt = gr.Button("Download CKPT")
                btn_to_pt = gr.Button("Convert to Diffusers PT")
                btn_push_pt = gr.Button("Push Diffusers PT to 🤗")
                btn_delete_pt = gr.Button("Delete Diffusers PT")
        btn_download_ckpt.click(download_ckpt, inputs=[text_ckpt_url], outputs=out_pt)
        btn_to_pt.click(to_pt, outputs=out_pt)
        btn_push_pt.click(push_pt, inputs=[text_pt_model_to, text_pt_token, text_pt_branch], outputs=out_pt)
        btn_delete_pt.click(delete_pt, outputs=out_pt)
    gr.Markdown(
    """
    ### ckpt to diffusers safetensors
    ckpt_url = <small>https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.ckpt or https://drive.google.com/file/d/file-id/view?usp=share_link or "https://civitai.com/api/download/models/5616?type=Model&format=PickleTensor"</small><br />
    safetensors_pt_model_to = camenduru/openjourney <br />
    branch = main <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_ckpt_to_safetensors_url = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_url")
                text_ckpt_to_safetensors_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_pt_model_to")
                text_ckpt_to_safetensors_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="safetensors_branch")
                text_ckpt_to_safetensors_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_ckpt_to_safetensors = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_download_ckpt_to_safetensors = gr.Button("Download CKPT")
                btn_ckpt_to_safetensors = gr.Button("Convert to Diffusers Safetensors")
                btn_push_ckpt_to_safetensors = gr.Button("Push Diffusers Safetensors to 🤗")
                btn_delete_ckpt_to_safetensors = gr.Button("Delete Diffusers Safetensors")
        btn_download_ckpt_to_safetensors.click(download_ckpt, inputs=[text_ckpt_to_safetensors_url], outputs=out_ckpt_to_safetensors)
        btn_ckpt_to_safetensors.click(from_ckpt_to_safetensors, outputs=out_ckpt_to_safetensors)
        btn_push_ckpt_to_safetensors.click(push_safetensors, inputs=[text_ckpt_to_safetensors_model_to, text_ckpt_to_safetensors_token, text_ckpt_to_safetensors_branch], outputs=out_ckpt_to_safetensors)
        btn_delete_ckpt_to_safetensors.click(delete_safetensors, outputs=out_ckpt_to_safetensors)
    gr.Markdown(
    """
    ### safetensors to diffusers pytorch
    safetensors_url = <small>https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.safetensors or https://drive.google.com/file/d/file-id/view?usp=share_link or "https://civitai.com/api/download/models/5616?type=Model&format=SafeTensor"</small><br />
    pt_model_to = camenduru/openjourney <br />
    branch = main <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_safetensors_to_pt_url = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_url")
                text_safetensors_to_pt_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="pt_model_to")
                text_safetensors_to_pt_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="pt_branch")
                text_safetensors_to_pt_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_safetensors_to_pt = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_download_safetensors_to_pt = gr.Button("Download Safetensors")
                btn_safetensors_to_pt = gr.Button("Convert to Diffusers PT")
                btn_push_safetensors_to_pt = gr.Button("Push Diffusers PT to 🤗")
                btn_delete_safetensors_to_pt = gr.Button("Delete Diffusers PT")
        btn_download_safetensors_to_pt.click(download_safetensors, inputs=[text_safetensors_to_pt_url], outputs=out_safetensors_to_pt)
        btn_safetensors_to_pt.click(from_safetensors_to_pt, outputs=out_safetensors_to_pt)
        btn_push_safetensors_to_pt.click(push_pt, inputs=[text_safetensors_to_pt_model_to, text_safetensors_to_pt_token, text_safetensors_to_pt_branch], outputs=out_safetensors_to_pt)
        btn_delete_safetensors_to_pt.click(delete_pt, outputs=out_safetensors_to_pt)
    gr.Markdown(
    """
    ### safetensors to diffusers safetensors
    safetensors_url = <small>https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.ckpt or https://drive.google.com/file/d/file-id/view?usp=share_link or "https://civitai.com/api/download/models/5616?type=Model&format=SafeTensor"</small><br />
    safetensors_model_to = camenduru/openjourney <br />
    branch = main <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_safetensors_to_safetensors_url = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_url")
                text_safetensors_to_safetensors_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_model_to")
                text_safetensors_to_safetensors_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="pt_branch")
                text_safetensors_to_safetensors_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_safetensors_to_safetensors = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_download_safetensors_to_safetensors = gr.Button("Download Safetensors")
                btn_safetensors_to_safetensors = gr.Button("Convert to Diffusers Safetensors")
                btn_push_safetensors_to_safetensors = gr.Button("Push Diffusers Safetensors to 🤗")
                btn_delete_safetensors_to_safetensors = gr.Button("Delete Diffusers Safetensors")
        btn_download_safetensors_to_safetensors.click(download_safetensors, inputs=[text_safetensors_to_safetensors_url], outputs=out_safetensors_to_safetensors)
        btn_safetensors_to_safetensors.click(from_safetensors_to_safetensors, outputs=out_safetensors_to_safetensors)
        btn_push_safetensors_to_safetensors.click(push_safetensors, inputs=[text_safetensors_to_safetensors_model_to, text_safetensors_to_safetensors_token, text_safetensors_to_safetensors_branch], outputs=out_safetensors_to_safetensors)
        btn_delete_safetensors_to_safetensors.click(delete_safetensors, outputs=out_safetensors_to_safetensors)
    gr.Markdown(
    """
    ### diffusers pytorch to diffusers flax <br />
    pt_model_from = dreamlike-art/dreamlike-diffusion-1.0 <br />
    flax_model_to = camenduru/dreamlike-diffusion-1.0 <br />
    branch = flax <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write <br />
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_pt_model_from = gr.Textbox(show_label=False, max_lines=1, placeholder="pt_model_from")
                text_flax_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="flax_model_to")
                text_flax_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="flax_branch")
                text_flax_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_flax = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_clone_pt = gr.Button("Clone Diffusers PT from 🤗")
                btn_to_flax = gr.Button("Convert to Diffusers Flax")
                btn_push_flax = gr.Button("Push Diffusers Flax to 🤗")
                btn_delete_flax = gr.Button("Delete Diffusers Flax")
        btn_clone_pt.click(clone_pt, inputs=[text_pt_model_from], outputs=out_flax)
        btn_to_flax.click(pt_to_flax, outputs=out_flax)
        btn_push_flax.click(push_flax, inputs=[text_flax_model_to, text_flax_token, text_flax_branch], outputs=out_flax)
        btn_delete_flax.click(delete_flax, outputs=out_flax)
    gr.Markdown(
    """
    ### diffusers flax to diffusers pytorch <br />
    flax_model_from = flax/mo-di-diffusion <br />
    pt_model_to =  camenduru/mo-di-diffusion <br />
    branch = pt <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write <br />
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_flax_model_from = gr.Textbox(show_label=False, max_lines=1, placeholder="flax_model_from")
                text_pt_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="pt_model_to")
                text_pt_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="pt_branch")
                text_pt_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_pt = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_clone_flax = gr.Button("Clone Diffusers Flax from 🤗")
                btn_to_pt = gr.Button("Convert to Diffusers PT")
                btn_push_pt = gr.Button("Push Diffusers PT to 🤗")
                btn_delete_pt = gr.Button("Delete Diffusers PT")
        btn_clone_flax.click(clone_flax, inputs=[text_flax_model_from], outputs=out_pt)
        btn_to_pt.click(flax_to_pt, outputs=out_pt)
        btn_push_pt.click(push_pt, inputs=[text_pt_model_to, text_pt_token, text_pt_branch], outputs=out_pt)
        btn_delete_pt.click(delete_pt, outputs=out_pt)
    gr.Markdown(
    """
    ### diffusers pytorch to ckpt
    pt_model_from = prompthero/openjourney <br />
    ckpt_name = openjourney <br />
    ckpt_model_to = camenduru/openjourney <br />
    branch = ckpt <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_pt_model_from = gr.Textbox(show_label=False, max_lines=1, placeholder="pt_model_from")
                text_ckpt_name = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_name")
                text_ckpt_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_model_to")
                text_ckpt_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="ckpt_branch")
                text_ckpt_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_ckpt = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_clone_pt = gr.Button("Clone Diffusers PT from 🤗")
                btn_to_ckpt = gr.Button("Convert to CKPT")
                btn_push_ckpt = gr.Button("Push CKPT to 🤗")
                btn_delete_ckpt = gr.Button("Delete CKPT")
        btn_clone_pt.click(clone_pt, inputs=[text_pt_model_from], outputs=out_ckpt)
        btn_to_ckpt.click(to_ckpt, inputs=[text_ckpt_name], outputs=out_ckpt)
        btn_push_ckpt.click(push_ckpt, inputs=[text_ckpt_model_to, text_ckpt_token, text_ckpt_branch], outputs=out_ckpt)
        btn_delete_ckpt.click(delete_ckpt, outputs=out_ckpt)
    gr.Markdown(
    """
    ### ckpt to safetensors <br />
    ckpt_url = <small>https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.ckpt or https://drive.google.com/file/d/file-id/view?usp=share_link or "https://civitai.com/api/download/models/5616?type=Model&format=PickleTensor"</small><br />
    safetensors_name = openjourney <br />
    safetensors_model_to = camenduru/openjourney <br />
    branch = safetensors <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write <br />
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_ckpt_url = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_url")
                text_safetensors_name = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_name")
                text_safetensors_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_model_to")
                text_safetensors_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="safetensors_branch")
                text_safetensors_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_safetensors = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_download_ckpt = gr.Button("Download CKPT")
                btn_to_safetensors = gr.Button("Convert to Safetensors")
                btn_push_safetensors = gr.Button("Push Safetensors to 🤗")
                btn_delete_safetensors = gr.Button("Delete Safetensors")
        btn_download_ckpt.click(download_ckpt, inputs=[text_ckpt_url], outputs=out_safetensors)
        btn_to_safetensors.click(to_safetensors, inputs=[text_safetensors_name], outputs=out_safetensors)
        btn_push_safetensors.click(push_safetensors, inputs=[text_safetensors_model_to, text_safetensors_token, text_safetensors_branch], outputs=out_safetensors)
        btn_delete_safetensors.click(delete_safetensors, outputs=out_safetensors)
    gr.Markdown(
    """
    ### safetensors to ckpt <br />
    safetensors_url = <small>https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.safetensors or https://drive.google.com/file/d/file-id/view?usp=share_link or "https://civitai.com/api/download/models/5616?type=Model&format=SafeTensor"</small><br />
    ckpt_name = openjourney <br />
    ckpt_model_to = camenduru/openjourney <br />
    branch = ckpt <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write <br />
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_safetensors_url = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_url")
                text_safetensors_to_ckpt_name = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_name")
                text_safetensors_to_ckpt_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_model_to")
                text_safetensors_to_ckpt_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="ckpt_branch")
                text_safetensors_to_ckpt_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_safetensors_to_ckpt = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_download_safetensors = gr.Button("Download Safetensors")
                btn_safetensors_to_ckpt = gr.Button("Convert to CKPT")
                btn_push_safetensors_to_ckpt = gr.Button("Push CKPT to 🤗")
                btn_delete_safetensors_ckpt = gr.Button("Delete CKPT")
        btn_download_safetensors.click(download_safetensors, inputs=[text_safetensors_url], outputs=out_safetensors_to_ckpt)
        btn_safetensors_to_ckpt.click(from_safetensors_to_ckpt, inputs=[text_safetensors_to_ckpt_name], outputs=out_safetensors_to_ckpt)
        btn_push_safetensors_to_ckpt.click(push_ckpt, inputs=[text_safetensors_to_ckpt_model_to, text_safetensors_to_ckpt_token, text_safetensors_to_ckpt_branch], outputs=out_safetensors_to_ckpt)
        btn_delete_safetensors_ckpt.click(delete_ckpt, outputs=out_safetensors_to_ckpt)
    gr.Markdown(
    """
    ### safetensors to safetensors emaonly <br />
    safetensors_url = <small>https://huggingface.co/ckpt/anything-v3.0/resolve/main/Anything-V3.0.safetensors or https://drive.google.com/file/d/file-id/view?usp=share_link or "https://civitai.com/api/download/models/4298?type=Model&format=SafeTensor"</small><br />
    emaonly_name = Anything-V3.0 <br />
    emaonly_model_to = camenduru/Anything-V3.0 <br />
    branch = safetensors <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write <br />
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_safetensors_url = gr.Textbox(show_label=False, max_lines=1, placeholder="safetensors_url")
                text_safetensors_to_emaonly_name = gr.Textbox(show_label=False, max_lines=1, placeholder="emaonly_name")
                text_safetensors_to_emaonly_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="emaonly_model_to")
                text_safetensors_to_emaonly_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="emaonly_branch")
                text_safetensors_to_emaonly_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_safetensors_to_emaonly = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_download_safetensors = gr.Button("Download Safetensors")
                btn_safetensors_to_emaonly = gr.Button("Convert to EMA Safetensors")
                btn_push_safetensors_to_emaonly = gr.Button("Push EMA Safetensors to 🤗")
                btn_delete_safetensors_emaonly = gr.Button("Delete EMA Safetensors")
        btn_download_safetensors.click(download_safetensors, inputs=[text_safetensors_url], outputs=out_safetensors_to_emaonly)
        btn_safetensors_to_emaonly.click(from_safetensors_to_emaonly, inputs=[text_safetensors_to_emaonly_name], outputs=out_safetensors_to_emaonly)
        btn_push_safetensors_to_emaonly.click(push_safetensors, inputs=[text_safetensors_to_emaonly_model_to, text_safetensors_to_emaonly_token, text_safetensors_to_emaonly_branch], outputs=out_safetensors_to_emaonly)
        btn_delete_safetensors_emaonly.click(delete_safetensors, outputs=out_safetensors_to_emaonly)
    gr.Markdown(
    """
    ### swap ckpt vae <br />
    ckpt_url = <small>https://huggingface.co/ckpt/anything-v3.0/resolve/main/Anything-V3.0-pruned.ckpt or https://drive.google.com/file/d/file-id/view?usp=share_link or "https://civitai.com/api/download/models/75?type=Model&format=PickleTensor"</small><br />
    vae_url = <small>https://huggingface.co/ckpt/anything-v3.0/resolve/main/Anything-V3.0.vae.pt or https://drive.google.com/file/d/file-id/view?usp=share_link or "https://civitai.com/api/download/models/5809?type=VAE&format=Other"</small><br />
    swaped_ckpt_name = Anything-V3.0 <br />
    swaped_ckpt_model_to = camenduru/Anything-V3.0 <br />
    swaped_ckpt_branch = ckpt <br />
    token = get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) new token role=write <br />
    """)
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text_ckpt_url = gr.Textbox(show_label=False, max_lines=1, placeholder="ckpt_url")
                text_vae_url = gr.Textbox(show_label=False, max_lines=1, placeholder="vae_url")
                text_swap_ckpt_name = gr.Textbox(show_label=False, max_lines=1, placeholder="swaped_ckpt_name")
                text_swap_ckpt_model_to = gr.Textbox(show_label=False, max_lines=1, placeholder="swaped_ckpt_model_to")
                text_swap_ckpt_branch = gr.Textbox(show_label=False, value="main", max_lines=1, placeholder="swaped_ckpt_branch")
                text_swap_ckpt_token = gr.Textbox(show_label=False, max_lines=1, placeholder="🤗 token")
                out_swap_ckpt = gr.Textbox(show_label=False)
            with gr.Row().style(equal_height=True):
                btn_download_ckpt = gr.Button("Download CKPT")
                btn_download_vae = gr.Button("Download VAE")
                btn_to_swap_ckpt = gr.Button("Swap CKPT VAE")
                btn_push_swap_ckpt = gr.Button("Push CKPT to 🤗")
                btn_delete_swap_ckpt = gr.Button("Delete CKPT")
        btn_download_ckpt.click(download_ckpt, inputs=[text_ckpt_url], outputs=out_swap_ckpt)
        btn_download_vae.click(download_vae, inputs=[text_vae_url], outputs=out_swap_ckpt)
        btn_to_swap_ckpt.click(swap_ckpt_vae, inputs=[text_swap_ckpt_name], outputs=out_swap_ckpt)
        btn_push_swap_ckpt.click(push_ckpt, inputs=[text_swap_ckpt_model_to, text_swap_ckpt_token, text_swap_ckpt_branch], outputs=out_swap_ckpt)
        btn_delete_swap_ckpt.click(delete_ckpt, outputs=out_swap_ckpt)

block.launch()