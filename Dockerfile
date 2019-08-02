FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential\
        curl \
        wget

WORKDIR /model

RUN wget https://deepai-opensource-codebases-models.s3-us-west-2.amazonaws.com/artbreeder/karras2019stylegan-ffhq-1024x1024.pkl

RUN pip3 install Flask==1.0.3 Pillow==6.1.0 requests==2.22.0 Flask-Cors==3.0.8 ai-integration==1.0.11



COPY server.py config.json ./
COPY dnnlib ./dnnlib

CMD []
ENTRYPOINT ["python3", "server.py"]
