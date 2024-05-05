#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import uvicorn
import numpy as np

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import CrossEncoder
from pydantic import BaseModel
from typing import Optional, List


# 环境变量传入
sk_key = os.environ.get('sk-key', 'sk-aaabbbcccdddeeefffggghhhiiijjjkkk')
MODEL_PATH = os.environ.get('model', '../bce-reranker-base_v1')

# 创建一个FastAPI实例
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建一个HTTPBearer实例
security = HTTPBearer()

class QADocs(BaseModel):
    query: Optional[str]
    documents: Optional[List[str]]


class Singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class ReRanker(metaclass=Singleton):
    def __init__(self, model_path):
        # 检测是否有GPU可用，如果有则使用cuda设备，否则使用cpu设备
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # if torch.cuda.is_available():
        #     print('本次加载模型的设备为GPU: ', torch.cuda.get_device_name(0))
        # else:
        #     print('本次加载模型的设备为CPU.')
        # 使用 Mac M1 GPU 参数为 mps
        # self.reranker = CrossEncoder(model_path, device="mps", max_length=512)
        self.reranker = CrossEncoder(model_path, device="mps")

    def compute_score(self, pairs: List[List[str]]):
        if len(pairs) > 0:
            scores = self.reranker.predict(pairs)
            if isinstance(scores, np.ndarray):
                # 转换 numpy 数组中的所有元素为基本 float 类型
                scores = scores.tolist()
            # 确保单个 float 也被转换
            # scores = [float(score) for score in scores]
            return scores
        else:
            return None


class Chat(object):
    def __init__(self, rerank_model_path: str = MODEL_PATH):
        self.reranker = ReRanker(rerank_model_path)

    def fit_query_answer_rerank(self, query_docs: QADocs) -> List:
        if query_docs is None or len(query_docs.documents) == 0:
            return []

        pair = [[query_docs.query, doc] for doc in query_docs.documents]
        scores = self.reranker.compute_score(pair)

        new_docs = []
        for index, score in enumerate(scores):
            new_docs.append({"index": index, "text": query_docs.documents[index], "score": score})
        results = [{"index": documents["index"], "relevance_score": documents["score"]} for documents in list(sorted(new_docs, key=lambda x: x["score"], reverse=True))]
        return results


@app.post('/v1/rerank')
async def handle_post_request(docs: QADocs, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != sk_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )
    chat = Chat()
    try:
        results = chat.fit_query_answer_rerank(docs)
        return {"results": results}
    except Exception as e:
        print(f"报错：\n{e}")
        return {"error": "重排出错"}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=6006)
