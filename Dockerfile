ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}
ARG PYTHON_IMAGE
ARG SOURCE_COMMIT=unknown
ARG SOURCE_SHA256
ENV QCENN_SOURCE_COMMIT=${SOURCE_COMMIT} \
    QCENN_EXPECTED_SOURCE_SHA256=${SOURCE_SHA256} \
    QCENN_BASE_IMAGE=${PYTHON_IMAGE} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md LICENSE CITATION.cff ./
COPY docker ./docker
COPY configs ./configs
COPY publication-evidence ./publication-evidence
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs
COPY .github ./.github
COPY Dockerfile docker-compose.yml requirements.lock .dockerignore .gitignore .gitattributes ./
RUN python -S docker/normalize_source.py /app
RUN python -m pip install --no-deps . && python -m pip check
ENV QCENN_REQUIRE_AER_PRELOAD=1
RUN python -S docker/entrypoint.py python -c "from qiskit_aer import AerSimulator; print(AerSimulator().name)"
ENTRYPOINT ["python", "-S", "/app/docker/entrypoint.py"]
CMD ["python", "-m", "qcenn_mi.validation", "--output-dir", "/artifacts", "--profile", "full"]
