FROM python:3.11-slim

# qiskit-aer and scikit-learn each bundle OpenMP; give glibc room for both
ENV GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096000
# that also eats the 2 MB default stack of Rust threads (Qiskit's transpiler)
ENV RUST_MIN_STACK=33554432

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# results and the downloaded dataset land in /app, mount it to keep them
VOLUME ["/app/results", "/app/data"]
ENTRYPOINT ["qudock"]
CMD ["--help"]
