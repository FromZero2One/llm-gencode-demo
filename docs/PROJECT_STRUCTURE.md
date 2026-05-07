# 项目结构说明

## 目录结构

```
llm-codegen-demo/
├── docs/                    # 文档目录
│   ├── README.md           # 主要文档
│   ├── QUICK_START.md      # 快速开始指南
│   ├── TEST_REPORT.md      # 测试报告
│   └── TOKENIZER_DEBUG_GUIDE.md  # Tokenizer调试指南
├── scripts/                # 脚本目录
│   ├── __init__.py         # Python包初始化文件
│   ├── main.py             # 主程序入口
│   ├── tokenizer.py        # Tokenizer实现
│   ├── attention.py        # 注意力机制实现
│   ├── transformer.py      # Transformer模型实现
│   ├── generator.py        # 代码生成器实现
│   ├── postprocessor.py    # 后处理器实现
│   ├── cache.py            # 缓存机制实现
│   ├── pipeline.py         # 完整管道实现
│   ├── visualizer.py       # 可视化工具实现
│   ├── test_all.py         # 完整测试套件
│   ├── test_performance.py # 性能测试
│   ├── debug_tokenizer.py  # Tokenizer调试工具
│   ├── tokenizer_interactive.py  # 交互式Tokenizer
│   └── tokenizer_pdb_debug.py    # PDB调试示例
├── run_demo.py             # 项目入口脚本
├── requirements.txt        # 依赖包列表
├── .gitignore              # Git忽略文件
└── attention_evolution_demo.png  # 演示图片
```

## 运行方式

### 1. 使用入口脚本（推荐）
```bash
python run_demo.py
```

### 2. 直接运行脚本
```bash
cd scripts
python test_all.py          # 运行测试
python main.py              # 运行主程序
python debug_tokenizer.py   # 运行调试工具
```

### 3. 从项目根目录运行
```bash
python -c "import sys; sys.path.insert(0, './scripts'); from tokenizer import SimpleTokenizer; print('Success')"
```

## 模块说明

- **docs/**: 存放所有文档文件，包括使用说明、测试报告等
- **scripts/**: 存放所有Python脚本文件，按功能组织
- **run_demo.py**: 统一的入口脚本，提供菜单式选择

## 注意事项

1. 所有脚本都在`scripts/`目录下，导入时会自动处理路径
2. 运行脚本时建议在项目根目录执行，以确保相对路径正确
3. 如需单独导入模块，请先将`scripts/`目录添加到Python路径中