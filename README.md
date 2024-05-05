# Mac 搭建私有AI知识库





## 环境

- Mac M1 Max 32G 内存：用于跑大模型

- NAS：在 Docker 中安装 FastGPT

## 大纲

1. **Ollama**：运行大模型，本次测试使用 qwen:14b 和 gemma:7b

2. **FastGPT**：基于 LLM 大模型的 AI 知识库构建平台

3. **OneAPI**：OpenAI 接口管理

4. **BCEmbedding**：开发Web服务运行`向量模型`和`二次重排模型`

   是由网易有道开发的中英双语和跨语种语义表征算法模型库，其中包含 `EmbeddingModel`和 `RerankerModel`两类基础模型。`EmbeddingModel`专门用于生成语义向量，在语义搜索和问答中起着关键作用，而 `RerankerModel`擅长优化语义搜索结果和语义相关顺序精排。

5. **模型下载**：从 [Huggingface 镜像站](https://hf-mirror.com/) 下载模型

## 安装配置

### 1. Ollama

- 软件下载：https://ollama.com/
- 模型下载：https://ollama.com/library
- 文档：https://github.com/ollama/ollama/blob/main/docs/faq.md
- 大模型：qwen:14b、qwen:7b、gemma:7b

- 运行状态：http://localhost:11434/

- Mac 配置外网访问

  ```sh
  launchctl setenv OLLAMA_HOST "0.0.0.0"
  ```

### 2. FastGPT

- 官网：https://fastgpt.run/
- 文档：https://doc.fastai.site/docs/intro/
- 默认用户名为`root`密码为`docker-compose.yml`环境变量里设置的 `DEFAULT_ROOT_PSW`
- 配置模型：https://doc.fastai.site/docs/development/configuration/

### 3. OneAPI

- 仓库：https://github.com/songquanpeng/one-api
- 默认账号为`root`密码为`123456`
- 添加 Ollama 渠道

### 4. BCEmbedding

- 文档：https://github.com/netease-youdao/BCEmbedding/blob/master/README_zh.md

- bce-embedding-base_v1 模型
- bce-reranker-base_v1 模型
- Web 服务开发

```sh
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 依赖安装
pip install -r requirements.txt

# 启动服务
python3 app.py

# 退出虚拟环境
deactivate
```

- Embedding 服务：
  - https://platform.openai.com/docs/api-reference/embeddings
- ReRanker 服务：
  - https://doc.fastai.site/docs/development/custom-models/bge-rerank/
  - https://doc.fastai.site/docs/development/configuration/#rerank-%E6%8E%A5%E5%85%A5

### 5. 模型下载

- 官方站：https://huggingface.co/

- 镜像站：https://hf-mirror.com/

1. 下载 hfd

   ```sh
   wget https://hf-mirror.com/hfd/hfd.sh
   chmod a+x hfd.sh
   ```

2. 设置环境变量

   ```sh
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. 安装 aria2c

   ```sh
   brew install aria2
   ```

4. 下载模型

   ```sh
   # 下载模型
   ./hfd.sh 模型名 --tool aria2c -x 下载线程数 --hf_username 用户名 --hf_token hf_xxx
   
   ./hfd.sh maidalun1020/bce-reranker-base_v1 --tool aria2c -x 5 --hf_username username --hf_token hf_xxx
   
   ./hfd.sh maidalun1020/bce-embedding-base_v1 --tool aria2c -x 5 --hf_username username --hf_token hf_xxx
   
   # 下载数据集
   ./hfd.sh wikitext --dataset --tool aria2c -x 4 --hf_username username --hf_token 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
   ```




测试：https://baike.baidu.com/item/%E8%8A%B1%E5%8D%89/229536#9