FROM fse-07-testing AS build

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python3 -m pip install "setuptools>=64" wheel
RUN python3 -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /src
