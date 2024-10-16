FROM nvcr.io/nvidia/pytorch:24.01-py3
RUN apt update && apt install -y python3-venv
RUN mkdir -p /opt
COPY ./requirements.txt /opt/requirements.txt
RUN python3 -m venv /opt/.venv &&  /opt/.venv/bin/pip3 install -r /opt/requirements.txt
WORKDIR /app
COPY . /app/
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
# ENTRYPOINT [ "/bin/bash" ] 
# CMD [ "/opt/.venv/bin/python3" ]
# ENTRYPOINT [ /opt/.venv/bin/python3 ]
