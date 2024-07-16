### Product Price Optimization

```bash
# 构建容器
cd ~/opt_price/docker

sudo docker build --network host -t priceopt:latest .


# 运行容器
cd ~/opt_price

sudo docker run --network host --name optprice -v $(pwd):/workspace  -it priceopt:latest

# 停止容器
# sudo docker stop optprice

# 删除容器
# sudo docker rm optprice

# docker exec 
# sudo docker exec -it optprice /bin/bash

## 进入到docker容器后
cd /workspace

python3
from amplpy import AMPL, modules
modules.activate("<license-uuid>")
ampl = AMPL()

# 运行服务
nohup python3 src/opt.py &

# 退出docker 并测试端口 control+d 退出
cd ~/opt_price

curl -X POST "http://127.0.0.1:52565/optimize" -H "Content-Type: application/json" -d @data/demo.json
```