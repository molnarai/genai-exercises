FROM nvcr.io/nvidia/pytorch:24.01-py3
WORKDIR /app
COPY . /app/
RUN python3 -m venv /app/.venv &&  /app/.venv/bin/pip3 install -r /app/requirements.txt
ENTRYPOINT [ "/bin/bash" ] 
CMD [ "/app/.venv/bin/python3" ]
