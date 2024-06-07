FROM python:3.8-slim-buster
#FROM anibali/pytorch:1.7.0-cuda11.0-ubuntu20.04
FROM anibali/pytorch:1.10.0-cuda11.3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
# Install system libraries required by OpenCV.
USER root

RUN apt-get -y update 
RUN apt-get -y install git
RUN mkdir /smart
RUN chown 1001:1001 /smart

WORKDIR /smart

USER 1001

RUN mkdir -p ./checkpoints/

ADD checkpoints/BERT  ./checkpoints/BERT

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY baselines.py  ./baselines.py
COPY losses.py ./losses.py
COPY data_loader.py  ./data_loader.py
COPY net.py ./net.py
COPY solve_VLAR.py ./solve_VLAR.py
COPY main.py	 ./main.py
COPY build_vocab.py	./build_vocab.py
COPY globvars.py	 ./globvars.py
COPY utils.py ./utils.py

ADD bakLlava-v1-hf ./checkpoints/bakLlava-v1-hf
ADD llava_dpt_1 ./checkpoints/llava_dpt_1
COPY data/icon-classes.txt  ./checkpoints/icon-classes.txt  
COPY data/SMART_info_v2.csv ./checkpoints/SMART_info_v2.csv


CMD ["python", "main.py", "--model_name", "resnet50", "--num_workers", "0", "--loss_type", "classifier", "--word_embed", "bert", "--split_type", "puzzle", "--challenge", "--phase", "val", "--pretrained_model_path", "./checkpoints/ckpt_resnet50_bert_212.pth"]


