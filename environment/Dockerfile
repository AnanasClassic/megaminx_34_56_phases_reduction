FROM golang:1.22.3-bookworm@sha256:f43c6f049f04cbbaeb28f0aad3eea15274a7d0a7899a617d0037aec48d7ab010

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git make p7zip-full python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements-training.txt /tmp/requirements-training.txt
RUN python3 -m venv /opt/mdr-venv \
    && /opt/mdr-venv/bin/pip install --no-cache-dir -r /tmp/requirements-training.txt
COPY . /work
ENV PATH=/opt/mdr-venv/bin:$PATH
ENV PYTHONPATH=/work/src

CMD ["make", "test"]
