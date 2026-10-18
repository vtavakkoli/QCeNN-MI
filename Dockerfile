ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}
ARG PYTHON_IMAGE
ARG SOURCE_COMMIT=unknown
ENV QCENN_SOURCE_COMMIT=${SOURCE_COMMIT} \
    QCENN_BASE_IMAGE=${PYTHON_IMAGE} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
      latexmk texlive-publishers texlive-latex-recommended \
      texlive-fonts-recommended poppler-utils libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md LICENSE CITATION.cff ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY paper ./paper
COPY docs ./docs
COPY .github ./.github
COPY Dockerfile docker-compose.yml requirements.lock .dockerignore .gitignore ./
RUN python -m pip install --no-deps . && python -m pip check
CMD ["python", "-m", "qcenn_mi.validation", "--output-dir", "/artifacts", "--publication"]
