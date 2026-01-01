FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_pytorch2

USER root
RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
RUN python3 -m pip install --upgrade pip


RUN pip3 install git+https://github.com/IDEA-Research/GroundingDINO.git
RUN pip3 install hf_xet 
RUN mkdir -p /tmp/GroundingDINO/config
RUN mkdir -p /tmp/GroundingDINO/weights

# Download GroundingDINO model weights
RUN curl -L https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/refs/heads/main/groundingdino/config/GroundingDINO_SwinT_OGC.py -o /tmp/GroundingDINO/config/GroundingDINO_SwinT_OGC.py
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O /tmp/GroundingDINO/weights/groundingdino_swint_ogc.pth


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/grounding-dino-adapter:0.1.7 -f Dockerfile .
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/grounding-dino-adapter:0.1.6
# docker push gcr.io/viewo-g/piper/agent/runner/apps/grounding-dino-adapter:0.1.7
