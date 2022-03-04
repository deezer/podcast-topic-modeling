FROM tomcat:9.0.30-jdk8-openjdk

LABEL maintainer="Francisco B. Valero <franbvalero@gmail.com>"
LABEL description="Docker image for evaluation"

RUN apt-get update --fix-missing && apt-get install -y \
    less \
    wget \
    python3 \
    python3-dev \
    python3-pip 

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas

WORKDIR /workspace
