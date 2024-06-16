### Product Price Optimization

```bash
# 构建容器
cd ~/opt_price/docker

sudo docker build --network host -t priceopt:latest .

sudo docker run --network host --name optprice -v $(pwd):/workspace  -it priceopt:latest

## 进入到docker容器后
cd /workspace/opt_price

# 运行服务
nohup python3 src/opt.py &

# 退出docker 并测试端口
cd ~/opt_price

curl -X POST "http://127.0.0.1:52565/optimize" -H "Content-Type: application/json" -d @data/demo.json
```