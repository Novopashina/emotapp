FROM python:3.7-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \  
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir tensorflow==2.3.0 opencv-python numpy

RUN pip install --no-cache-dir "protobuf<=3.19.0"

RUN pip install pypaz --user

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /code

COPY . .

EXPOSE 8081

CMD ["uvicorn", "emotapp:app", "--host", "0.0.0.0", "--port", "8081"]
