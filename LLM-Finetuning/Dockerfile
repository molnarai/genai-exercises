FROM nvcr.io/nvidia/pytorch:24.01-py3
WORKDIR /app
# RUN apt update && apt install -y python3-venv
RUN mkdir -p /opt
# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

COPY ./requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt

# ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
# ENTRYPOINT [ "/bin/bash" ] 
# CMD [ "/opt/.venv/bin/python3" ]
# ENTRYPOINT [ /opt/.venv/bin/python3 ]
