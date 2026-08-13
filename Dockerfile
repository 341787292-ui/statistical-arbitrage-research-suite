FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY quant_research_agent ./quant_research_agent
COPY samples ./samples

ENTRYPOINT ["python", "main.py"]
CMD ["--paper", "samples/stat_arb_note.txt", "--no-llm", "--run-agent"]
