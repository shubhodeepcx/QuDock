FROM python:3.11-slim

# qiskit-aer and scikit-learn each bundle OpenMP; give glibc room for both
ENV GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096000
# that also eats the 2 MB default stack of Rust threads (Qiskit's transpiler)
ENV RUST_MIN_STACK=33554432
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 1000 qudock \
    && mkdir -p /app/results /app/data \
    && chown -R qudock /app

# no reason for the studies to run as root
USER qudock

# results and the downloaded dataset land in /app, mount it to keep them
VOLUME ["/app/results", "/app/data"]
ENTRYPOINT ["qudock"]
CMD ["--help"]
