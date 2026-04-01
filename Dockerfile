FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install --upgrade pip
RUN pip install openenv-core fastapi uvicorn pydantic
EXPOSE 7860
CMD ["python", "server/app.py"]