# 基于焊接工艺参数的机器人焊接缺陷智能预测系统

制造智能技术课程设计项目，运用 vibe coding 方法开发的 B/S 架构焊接质量智能预测应用。

## 项目简介

面向汽车制造、航空航天等行业的机器人焊接质量监控场景，工艺人员输入7项关键焊接工艺参数，系统自动输出缺陷预测结果、缺陷概率及各参数影响程度分析，辅助工艺人员及时调整参数。

## 技术方向覆盖（3个课程专题）

| 技术方向 | 在系统中的作用 |
|---------|--------------|
| 数据预处理与特征工程 | 对7个工艺参数进行 StandardScaler 标准化，消除量纲差异，按7:3分层抽样划分数据集 |
| 监督学习（分类） | 随机森林分类模型，根据工艺参数预测是否存在缺陷，输出缺陷概率 |
| 数据可视化 | ECharts 绘制特征重要性排序图、历史预测分布饼图、概率进度条 |

## 技术栈

- **前端**：HTML5 + CSS3 + JavaScript + ECharts 5
- **后端**：Python + Flask
- **数据库**：SQLite 3
- **算法库**：scikit-learn、pandas、numpy、joblib
- **架构**：B/S 架构

## 目录结构

```
dxb课程设计/
├── app.py                      # Flask 主程序（API接口 + 页面路由）
├── requirements.txt            # Python 依赖清单
├── readme.md                   # 项目说明文档
├── 选题说明.md                  # 选题说明
├── 方案设计.md                  # 详细方案设计
├── 学习笔记.md                  # vibe coding 学习笔记
├── algorithm/
│   ├── preprocess.py           # 数据预处理脚本
│   └── train.py                # 模型训练脚本
├── database/
│   ├── __init__.py
│   └── db.py                   # SQLite 数据库操作
├── model/
│   ├── classifier.pkl          # 随机森林分类模型
│   ├── scaler.pkl              # StandardScaler 标准化器
│   ├── feature_map.json        # 特征名映射
│   ├── feature_importance.json # 特征重要性
│   └── metrics.json            # 模型评估指标
├── data/
│   ├── 1_Robotic_welding_edge_dataset_v2.csv  # 原始数据集
│   ├── welding_data_processed.csv             # 标准化后数据
│   ├── X_train.csv / y_train.csv              # 训练集
│   └── X_test.csv / y_test.csv                # 测试集
├── templates/
│   └── index.html              # 前端页面
├── static/
│   ├── css/style.css           # 页面样式
│   └── js/main.js              # 前端交互逻辑
├── tests/
│   └── test_app.py             # 自动化测试脚本
└── prompt/                     # AI 对话过程档案
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理（可选，数据已预处理完成）

```bash
python algorithm/preprocess.py
```

### 3. 训练模型（可选，模型已训练完成）

```bash
python algorithm/train.py
```

### 4. 启动系统

```bash
python app.py
```

启动后访问：**http://127.0.0.1:5000**

### 5. 运行自动化测试

```bash
python tests/test_app.py
```

## API 接口文档

### 预测接口
- **URL**: `POST /api/predict`
- **Content-Type**: `application/json`
- **请求参数**:

| 参数名 | 类型 | 说明 | 范围 |
|-------|------|------|------|
| arc_voltage | float | 电弧电压(V) | 0-50 |
| welding_current | float | 焊接电流(A) | 0-500 |
| welding_speed | float | 焊接速度(cm/min) | 0-100 |
| wire_feed_speed | float | 送丝速度(m/min) | 0-30 |
| gas_flow_rate | float | 气体流量(L/min) | 0-50 |
| torch_angle | float | 焊枪角度(°) | 0-90 |
| base_metal_temperature | float | 母材温度(°C) | 0-200 |

- **响应示例**:
```json
{
  "success": true,
  "prediction": 0,
  "prediction_label": "无缺陷",
  "defect_probability": 0.11,
  "normal_probability": 0.89,
  "confidence": 0.89,
  "message": "焊接参数正常，预测无缺陷",
  "status": "normal"
}
```

### 历史记录
- **URL**: `GET /api/history?limit=50`
- **功能**: 获取历史预测记录，按时间倒序

### 特征重要性
- **URL**: `GET /api/importance`
- **功能**: 获取7个工艺参数的特征重要性排序

### 统计信息
- **URL**: `GET /api/statistics`
- **功能**: 获取预测统计和模型评估指标

### 健康检查
- **URL**: `GET /api/health`
- **功能**: 检查服务和模型加载状态

## 模型性能

| 指标 | 值 |
|------|-----|
| 准确率 (Accuracy) | 62.22% |
| 精确率 (Precision) | 60.42% |
| 召回率 (Recall) | 65.91% |
| F1-Score | 63.04% |
| 训练样本 | 210条 |
| 测试样本 | 90条 |

## 数据来源

Mendeley Data 平台公开发布的 "Simulated Dataset for Edge-Based Defect Prediction in Robotic Welding"。

- 发布信息：2025年7月16日发布，Version 1
- DOI: 10.17632/ndcns86bzt.1
- 下载地址：https://data.mendeley.com/datasets/ndcns86bzt/1
- 数据规模：300个样本
- 说明：该数据集为模拟数据集，基于焊接工艺参数阈值规则生成，仅用于学术研究

## 系统功能

1. **参数输入**：7项焊接工艺参数数值输入，支持快捷填充正常/异常示例
2. **缺陷预测**：随机森林模型输出是否存在缺陷及缺陷概率
3. **影响分析**：特征重要性排序柱状图，展示各参数对质量的影响程度
4. **结果可视化**：缺陷/正常概率进度条、历史预测分布饼图
5. **历史记录**：所有预测记录持久化存储，表格展示
6. **统计概览**：总预测数、正常数、缺陷数、缺陷率、模型准确率实时展示
