FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    xformers==0.0.28.post1 omegaconf==2.2.3 open3d==0.18.0 plyfile==1.0.3 pillow==9.5.0 ftfy==6.2.0 diffdist==0.1 gsplat==1.0.0 torchmetrics==1.3.2 torchsde==0.2.6 \
    timm==0.9.2 wandb==0.17.4 regex==2024.9.11 einops==0.4.1 transformers==4.42.3 diffusers==0.25.1 accelerate==0.21.0 huggingface_hub==0.24.7 \
    https://github.com/camenduru/wheels/releases/download/colab4/detectron2-0.6-cp310-cp310-linux_x86_64.whl \
    https://github.com/camenduru/wheels/releases/download/3090/natten-0.14.6-cp310-cp310-linux_x86_64.whl \
    https://github.com/camenduru/wheels/releases/download/colab4/MultiScaleDeformableAttention-1.0-cp310-cp310-linux_x86_64.whl \
    https://github.com/camenduru/wheels/releases/download/3090/gsplat-1.0.0-cp310-cp310-linux_x86_64.whl && \
    git clone -b dev https://github.com/camenduru/VistaDream /content/VistaDream && \
    pip install -q /content/VistaDream/tools/DepthPro && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/bakLlava/added_tokens.json -d /content/VistaDream/tools/bakLlava -o added_tokens.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/bakLlava/chat_template.json -d /content/VistaDream/tools/bakLlava -o chat_template.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/bakLlava/config.json -d /content/VistaDream/tools/bakLlava -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/bakLlava/generation_config.json -d /content/VistaDream/tools/bakLlava -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/bakLlava/model-00001-of-00004.safetensors -d /content/VistaDream/tools/bakLlava -o model-00001-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/bakLlava/model-00002-of-00004.safetensors -d /content/VistaDream/tools/bakLlava -o model-00002-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/bakLlava/model-00003-of-00004.safetensors -d /content/VistaDream/tools/bakLlava -o model-00003-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/bakLlava/model-00004-of-00004.safetensors -d /content/VistaDream/tools/bakLlava -o model-00004-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/bakLlava/model.safetensors.index.json -d /content/VistaDream/tools/bakLlava -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/bakLlava/preprocessor_config.json -d /content/VistaDream/tools/bakLlava -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/bakLlava/special_tokens_map.json -d /content/VistaDream/tools/bakLlava -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/bakLlava/tokenizer.json -d /content/VistaDream/tools/bakLlava -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/bakLlava/tokenizer.model -d /content/VistaDream/tools/bakLlava -o tokenizer.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/bakLlava/tokenizer_config.json -d /content/VistaDream/tools/bakLlava -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/feature_extractor/preprocessor_config.json -d /content/VistaDream/tools/StableDiffusion/ckpt/feature_extractor -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/safety_checker/config.json -d /content/VistaDream/tools/StableDiffusion/ckpt/safety_checker -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/sd15/safety_checker/pytorch_model.bin -d /content/VistaDream/tools/StableDiffusion/ckpt/safety_checker -o pytorch_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/scheduler/scheduler_config.json -d /content/VistaDream/tools/StableDiffusion/ckpt/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/text_encoder/config.json -d /content/VistaDream/tools/StableDiffusion/ckpt/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/sd15/text_encoder/pytorch_model.bin -d /content/VistaDream/tools/StableDiffusion/ckpt/text_encoder -o pytorch_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/tokenizer/merges.txt -d /content/VistaDream/tools/StableDiffusion/ckpt/tokenizer -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/tokenizer/special_tokens_map.json -d /content/VistaDream/tools/StableDiffusion/ckpt/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/tokenizer/tokenizer_config.json -d /content/VistaDream/tools/StableDiffusion/ckpt/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/tokenizer/vocab.json -d /content/VistaDream/tools/StableDiffusion/ckpt/tokenizer -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/unet/config.json -d /content/VistaDream/tools/StableDiffusion/ckpt/unet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/sd15/unet/diffusion_pytorch_model.bin -d /content/VistaDream/tools/StableDiffusion/ckpt/unet -o diffusion_pytorch_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/vae/config.json -d /content/VistaDream/tools/StableDiffusion/ckpt/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/sd15/vae/diffusion_pytorch_model.bin -d /content/VistaDream/tools/StableDiffusion/ckpt/vae -o diffusion_pytorch_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/raw/main/sd15/model_index.json -d /content/VistaDream/tools/StableDiffusion/ckpt -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/sd_xl_offset_example-lora_1.0.safetensors -d /content/VistaDream/tools/Fooocus/models/loras -o sd_xl_offset_example-lora_1.0.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/pytorch_model.bin -d /content/VistaDream/tools/Fooocus/models/prompt_expansion/fooocus_expansion -o pytorch_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/pytorch_lora_weights.safetensors -d /content/VistaDream/tools/StableDiffusion/lcm_ckpt -o pytorch_lora_weights.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/juggernautXL_v8Rundiffusion.safetensors -d /content/VistaDream/tools/Fooocus/models/checkpoints -o juggernautXL_v8Rundiffusion.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/inpaint_v26.fooocus.patch -d /content/VistaDream/tools/Fooocus/models/inpaint -o inpaint_v26.fooocus.patch && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/fooocus_upscaler_s409985e5.bin -d /content/VistaDream/tools/Fooocus/models/unscale_models -o fooocus_upscaler_s409985e5.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/depth_pro.pt -d /content/VistaDream/tools/DepthPro/checkpoints -o depth_pro.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/VistaDream/resolve/main/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth -d /content/VistaDream/tools/OneFormer/ckpts -o coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth

COPY ./worker_runpod.py /content/VistaDream/worker_runpod.py
WORKDIR /content/VistaDream
CMD python worker_runpod.py