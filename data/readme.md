## 数据来源

本系统采用 Mendeley Data 平台公开发布的"Simulated Dataset for Edge-Based Defect Prediction in Robotic Welding"。

- 发布信息：2025年7月16日发布，Version 1
- DOI: 10.17632/ndcns86bzt.1
- 下载地址： https://data.mendeley.com/datasets/ndcns86bzt/1
- 数据规模：300个样本
- 特征数量：7个（电弧电压、焊接电流、焊接速度、送丝速度、气体流量、焊枪角度、母材温度）
- 标签：defect_label（0=无缺陷，1=有缺陷）
- 说明：该数据集为模拟数据集，基于焊接工艺参数阈值规则生成，仅用于学术研究



## 数据预处理

### 预处理步骤
1. 加载原始数据（支持 .csv 和 .xlsx 格式）
2. 检查缺失值和数据类型
3. 分离特征（7个焊接参数）和标签（defect_label）
4. 使用 StandardScaler 进行标准化
5. 按 7:3 划分训练集和测试集（分层抽样）
6. 保存标准化器和处理后的数据文件

### 预处理结果
| 文件 | 说明 |
|------|------|
| welding_data_processed.csv | 标准化后的完整数据 |
| X_train.csv / y_train.csv | 训练集（210条） |
| X_test.csv / y_test.csv | 测试集（90条） |

### 运行预处理
```bash
python algorithm/preprocess.py