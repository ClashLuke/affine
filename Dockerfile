FROM python:3.11-slim

RUN pip install uv

WORKDIR /app
COPY pyproject.toml .
COPY affine/ affine/
RUN uv pip install --system .

ENTRYPOINT ["affine"]
