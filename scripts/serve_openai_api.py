"""
OpenAI 兼容 API 服务 - 将本项目的代码生成模型暴露为标准 LLM API

参考 minimind/scripts/serve_openai_api.py 的设计，但针对本项目特点简化：
- 无 chat template：直接把 messages 拼接成 prompt 文本
- 无 thinking/tool-calls 解析：本项目是教学模型，输出即普通文本
- 保留核心：/v1/chat/completions（流式 SSE + 非流式）、/v1/models

使用方法：
    python scripts/serve_openai_api.py --preset tiny --port 8000

测试：
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "llm-codegen-demo", "messages": [{"role": "user", "content": "public class"}], "stream": false}'

兼容 OpenAI SDK：
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
    resp = client.chat.completions.create(
        model="llm-codegen-demo",
        messages=[{"role": "user", "content": "public class"}]
    )
"""

import argparse
import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scripts.pipeline import CodeGenerationPipeline

app = FastAPI(title="LLM CodeGen Demo API", version="1.0.0")

# 全局 pipeline（启动时初始化）
_pipeline = None
_model_name = "llm-codegen-demo"


class ChatMessage(BaseModel):
    """OpenAI 消息格式：{'role': 'system'|'user'|'assistant', 'content': '...'}"""
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    """OpenAI chat.completions 请求体（字段与官方 API 对齐，采样参数映射到本项目策略）"""
    model: str = "llm-codegen-demo"
    messages: list
    temperature: float = Field(default=0.7, ge=0.0, allow_inf_nan=False)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=50, ge=1)
    stream: bool = False
    # OpenAI 生态常见参数（部分映射到本项目采样策略）
    top_k: int = Field(default=50, ge=1)


def init_pipeline(args):
    """初始化 pipeline（支持加载训练好的 checkpoint）"""
    global _pipeline, _model_name
    _pipeline = CodeGenerationPipeline(
        preset=args.preset,
        device=args.device,
        architecture=args.architecture,
    )
    if args.checkpoint:
        _pipeline.load_checkpoint(args.checkpoint)
    _model_name = args.model_name


def messages_to_prompt(messages: list) -> str:
    """
    把 OpenAI messages 列表拼接为单条 prompt

    本项目没有 chat template，采用最简单的角色前缀拼接：
    system/user 的内容直接连接，assistant 内容忽略（教学模型不处理多轮历史）
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("system", "user"):
            parts.append(content)
    return "\n".join(parts) if parts else "public class"


def make_chunk(chunk_id: str, delta: dict, finish_reason=None) -> str:
    """构造一个 OpenAI 流式 chunk（SSE data 载荷）"""
    return json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }, ensure_ascii=False)


def generate_stream(prompt: str, req: ChatRequest):
    """
    流式生成（SSE）

    本项目的 CodeGenerator 逐 token 生成，这里把每个 token 作为一个
    OpenAI chunk 推送。用生成器的惰性特性实现"边生成边返回"。
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    try:
        # 首个 chunk：role 声明（OpenAI 惯例）
        yield f"data: {make_chunk(chunk_id, {'role': 'assistant'})}\n\n"

        result = _pipeline.generator.generate(
            prompt=prompt,
            max_length=req.max_tokens,
            temperature=req.temperature,
            verbose=False,
            strategy='top_p' if req.top_p < 1.0 else 'temperature',
            top_k=req.top_k,
            top_p=req.top_p,
        )
        # 教学简化：整段文本作为一个 content chunk 推出
        # （逐 token 流式需要重构 generate 为生成器，后续优化项）
        text = result['code']
        for i in range(0, len(text), 8):
            yield f"data: {make_chunk(chunk_id, {'content': text[i:i+8]})}\n\n"

        yield f"data: {make_chunk(chunk_id, {}, finish_reason='stop')}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    """OpenAI 兼容的对话补全接口"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="模型尚未初始化")

    try:
        prompt = messages_to_prompt(request.messages)

        if request.stream:
            return StreamingResponse(
                generate_stream(prompt, request),
                media_type="text/event-stream",
            )

        # 非流式：走 pipeline 完整流程（含缓存）
        result = _pipeline.generate(
            prompt=prompt,
            max_length=request.max_tokens,
            temperature=request.temperature,
            verbose=False,
            strategy='top_p' if request.top_p < 1.0 else 'temperature',
            top_k=request.top_k,
            top_p=request.top_p,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result['processed_code']},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,  # 教学模型未精确统计
                "completion_tokens": result['token_count'],
                "total_tokens": result['token_count'],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
def list_models():
    """OpenAI 兼容的模型列表接口"""
    return {
        "object": "list",
        "data": [{
            "id": _model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llm-codegen-demo",
        }],
    }


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok", "model_loaded": _pipeline is not None}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="LLM CodeGen Demo OpenAI API Server")
    parser.add_argument('--preset', default='tiny', type=str,
                        choices=['tiny', 'small', 'medium'], help='模型预设')
    parser.add_argument('--architecture', default='encdec', type=str,
                        choices=['encdec', 'decoder'], help='模型架构')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='加载训练好的 checkpoint 路径(.pt)')
    parser.add_argument('--device', default='cpu', type=str, help='计算设备')
    parser.add_argument('--host', default='0.0.0.0', type=str, help='监听地址')
    parser.add_argument('--port', default=8000, type=int, help='监听端口')
    parser.add_argument('--model_name', default='llm-codegen-demo', type=str,
                        help='对外暴露的模型名')
    args = parser.parse_args()

    init_pipeline(args)

    uvicorn.run(app, host=args.host, port=args.port)
