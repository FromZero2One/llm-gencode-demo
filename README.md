# LLM代码生成演示项目

这个项目用于理解和调试大模型代码生成的完整过程。

## 项目结构

```
llm-codegen-demo/
├── tokenizer.py          # Token化模块
├── transformer.py        # Transformer模型核心
├── attention.py          # 注意力机制
├── generator.py          # 代码生成器
├── postprocessor.py      # 后处理模块
├── cache.py              # 缓存机制
├── pipeline.py           # 完整流程管道
├── visualizer.py         # 可视化工具
├── main.py               # 主程序入口
└── requirements.txt      # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python main.py
```

## 功能说明

1. **Token化**: 将文本转换为token序列
2. **位置编码**: 添加位置信息
3. **注意力机制**: 计算token间的关系
4. **模型推理**: 前向传播获取概率分布
5. **采样策略**: Top-k、Temperature采样
6. **代码生成**: 逐步生成token
7. **后处理**: 格式化、验证、优化
8. **缓存**: 相似请求的缓存机制
9. **可视化**: 注意力权重、生成过程可视化

## 调试技巧

- 设置 `DEBUG=True` 查看详细日志
- 使用 `visualizer.py` 查看注意力权重
- 修改 `temperature` 和 `top_k` 观察生成效果
