FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

LABEL maintainer="Francisco B. Valero <franbvalero@gmail.com>"
LABEL description="Docker image for experimentation"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    libsqlite3-0 \
    libsqlite3-dev \
    sqlite3 \
    tar \
    wget \
    less \
    vim \
    python3-pip \
    g++ \
    build-essential \
    make \
    protobuf-compiler \
    libprotobuf-dev

RUN pip install --upgrade pip
RUN pip install git+https://github.com/informagi/REL
RUN mkdir -p /.rel_cache /.flair
RUN chmod 777 /.rel_cache /.flair

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN mkdir -p /usr/local/share/nltk_data
RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); import nltk; nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/local/share/nltk_data')"]

WORKDIR /workspace
