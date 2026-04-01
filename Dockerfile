FROM python:3.10

WORKDIR /app

# Copy all project files
COPY . .

# Install required packages
RUN pip install --upgrade pip
RUN pip install openenv-core fastapi uvicorn pydantic pandas numpy

# HF Spaces must use port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "server/app.py"]