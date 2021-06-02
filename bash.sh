#!/bin/bash

# install packages
conda install -c conda-forge ffmpeg -y
conda install -c conda-forge scikit-video -y
pip install -r requirements.txt
python -m pip install nltk
python -m nltk.downloader punkt

# download pretrained models
FILE="data/svqad/svqad-qa_vocab.json"
if [[ -f "$FILE" ]]; 
then
    echo $FILE existed
else
	mkdir -p data/svqad/
	wget -O svqad-qa_vocab.json --no-check-certificate "https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712786_student_hcmus_edu_vn/EQEgiCfrSg1AmDDGSwx6_LsBhvrM2TtcH22QvBRQxoIz-Q?e=xNZgz1\&download=1"
	mv svqad-qa_vocab.json data/svqad/
fi

FILE="results/expSVQAD-QA/ckpt/model.pt"
if [[ -f "$FILE" ]]; 
then
    echo $FILE existed
else
	mkdir -p results/expSVQAD-QA/ckpt/
	wget -O svqad_model.pt --no-check-certificate "https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712786_student_hcmus_edu_vn/EcYzm1adcw1DsBhofKH_PgkB1ng5s3Q43fAEkEhxeoFRGQ?e=mh7LqG\&download=1"
	mv svqad_model.pt results/expSVQAD-QA/ckpt/model.pt
fi

FILE="data/preprocess/pretrained/resnext-101-kinetics.pth"
if [[ -f "$FILE" ]]; 
then
    echo $FILE existed
else
	mkdir -p data/preprocess/pretrained
	wget -O resnext-101-kinetics.pth --no-check-certificate "https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/1712786_student_hcmus_edu_vn/EdhtNUaJQ-dGrCLXxS-YloYB0bumQU8MzuSWXGM0CpZW3g?e=sYa6yX\&download=1"
	mv resnext-101-kinetics.pth data/preprocess/pretrained
fi

python prepare_pretrained_model.py
