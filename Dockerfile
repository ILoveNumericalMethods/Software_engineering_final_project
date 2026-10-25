FROM fse-07-testing AS build

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python3 -m pip install "setuptools>=64" wheel
RUN python3 -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

RUN python3 --version
RUN c++ --version
RUN python3 -c "import torch; print('PyTorch:', torch.__version__); assert torch.version.cuda is None"

WORKDIR /src
COPY . .
RUN python3 -m pip wheel . --no-deps --no-build-isolation -w /wheels
RUN ls /wheels/*.whl

FROM fse-07-testing AS runtime

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python3 -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install pytest

RUN python3 --version
RUN python3 -c "import torch; print('PyTorch:', torch.__version__); assert torch.version.cuda is None"
RUN python3 -m pytest --version

COPY --from=build /wheels /wheels
RUN python3 -m pip install /wheels/*.whl && python3 -m pip show fse-ops
RUN python3 -c "import fse_ops; print(fse_ops.__file__)"
COPY tests /opt/tests

WORKDIR /opt
CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]
