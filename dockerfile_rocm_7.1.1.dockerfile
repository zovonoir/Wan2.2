FROM ubuntu:22.04

RUN mkdir -p /app
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx1100;gfx1101;gfx1200;gfx1201;gfx950
ENV PATH=/opt/rocm/llvm/bin:$PATH
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:

ARG PYTORCH_REPO="https://github.com/pytorch/pytorch"
ARG PYTORCH_VISION_REPO="https://github.com/pytorch/vision.git"
ARG KINETO_REPO="https://github.com/mwootton/kineto"
ARG KINETO_BRANCH="f624d5d77ef1998f763d3dc969a43bc85b39dbea"
ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG IRIS_REPO="https://github.com/ROCm/iris.git"
ARG IRIS_BRANCH="0dfc460e37517c05ab88f1b348d491ae30abb673"
ARG FLASH_ATTENTION_REPO="https://github.com/Dao-AILab/flash-attention.git"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG AITER_BRANCH="a6652a7790627ec769aa42f56da1d51001e63f98"

# install rocm 7.1.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgfortran5 libgl1 libglib2.0-dev liblzma-dev \
    ninja-build python3 python3-dev python3-packaging python3-pip \
    python3-venv wget vim git rsync dialog \
    && wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/jammy/amdgpu-install_7.1.1.70101-1_all.deb \
    && apt install -y ./amdgpu-install_7.1.1.70101-1_all.deb \
    && apt-get update && apt-get install -y rocm && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN groupadd -r -f render && groupadd -r -f video && usermod -a -G render,video root
RUN ln -s python3 /usr/bin/python

# install python wheel
RUN pip3 install --upgrade pip && apt install -y python3-setuptools python3-wheel && pip3 install setuptools[core]

# WORKDIR /app/pytorch
RUN git clone ${PYTORCH_REPO} pytorch \
    && cd pytorch \
    && pip install -r requirements.txt \
    && git submodule update --init --recursive \
    && git -C ./third_party/kineto remote set-url "origin" ${KINETO_REPO} \
    && git -C ./third_party/kineto fetch \
    && sed -i '287s/maxBufferSize_{1000000}/maxBufferSize_{100000000}/g' ./third_party/kineto/libkineto/src/RoctracerLogger.h \
    && pip uninstall setuptools -y && pip install setuptools[core] \
    && python3 tools/amd_build/build_amd.py \
    && CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)'):/opt/rocm-7.1.1 BUILD_TEST=0 python3 setup.py bdist_wheel --dist-dir=dist \
    && pip3 install /app/pytorch/dist/*.whl \
    && cd .. \
    && rm -rf pytorch

WORKDIR /app/vision
RUN git clone ${PYTORCH_VISION_REPO} vision \
    && cd vision \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && pip3 install dist/*.whl \
    && cd .. \
    && rm -rf vision

RUN pip3 install --no-cache-dir "opencv-python>=4.9.0.80" \
                                "diffusers>=0.31.0" \
                                "transformers>=4.49.0,<=4.51.3" \
                                "tokenizers==0.21.4" \
                                "accelerate>=1.1.1" \
                                "tqdm==4.67.1" \
                                "imageio[ffmpeg]" \
                                "easydict" \
                                "ftfy==6.3.1" \
                                "dashscope==1.25.0" \
                                "imageio-ffmpeg" \
                                "numpy>=1.23.5,<2" \
                                "decord==0.6.0" \
                                "librosa==0.11.0" \
                                "peft==0.17.1" \
                                ninja

WORKDIR /app
RUN git clone ${TRITON_REPO} triton \
    && cd triton \
    && pip3 install -r python/requirements.txt \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && pip3 install /app/triton/dist/*.whl \
    && cd .. \
    && rm -rf triton

RUN git clone ${IRIS_REPO} iris && cd iris && git checkout ${IRIS_BRANCH} && pip install -e .

RUN git clone ${FLASH_ATTENTION_REPO} flash-attention \
    && cd flash-attention \
    && git submodule update --init \
    && GPU_ARCHS=$(echo ${PYTORCH_ROCM_ARCH} | sed -e 's/;gfx1[0-9]\{3\}//g') python3 setup.py bdist_wheel --dist-dir=dist \
    && pip3 install /app/flash-attention/dist/*.whl \
    && cd .. \
    && rm -rf flash-attention

RUN git clone https://github.com/zovonoir/Wan2.2.git && cd Wan2.2 && git checkout i2v_release/triton_kernel_optimized1 && git pull
RUN mkdir -p /app/Wan2.2/generate_video

RUN git clone ${AITER_REPO} aiter \
    && cd aiter \
    && git checkout ${AITER_BRANCH}  \
    && git submodule sync \
    && git submodule update --init --recursive \
    && python3 setup.py develop

WORKDIR /app