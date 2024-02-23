# 1. Base image
FROM python:3.8

# 2. Specify directory
WORKDIR /root

# 3. Copy files
COPY src /root/src

COPY requirements.txt /root/requirements.txt

# 4. Install dependencies
RUN pip install -r /root/requirements.txt

# 5. For run main file
RUN ["python","root/main.py"]
