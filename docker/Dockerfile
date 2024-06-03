# docker build --network host -t priceopt:latest .

FROM ddookkee/ipopt:3.14

# 设置工作目录和user
WORKDIR /workspace
USER root

# # 基本环境
# # RUN apt update && \
# #     apt install git
# RUN apt update && \
#     apt-get install nano

RUN apt-get install python3.10 -y &&\
    rm /usr/bin/python3 &&\
    ln -s python3.10 /usr/bin/python3 &&\ 
    apt update &&\
    apt-get install curl -y &&\
    curl -sS https://bootstrap.pypa.io/get-pip.py

# --- python pakgs ---
# 复制本地文件到容器中  
COPY requirements.txt .

# # 安装Python依赖z 
RUN apt-get install -y --no-install-recommends \
            python3-dev \
            python3-pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# docker run --network host --name optprice -v $(pwd):/workspace  -it priceopt:latest