"""
Flask 后端服务
机器人焊接缺陷智能预测系统
功能：提供焊接缺陷预测API、历史记录查询、特征重要性查询、统计信息
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# 将项目根目录加入路径，确保能导入 database 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db import init_db, insert_prediction, get_history, get_statistics

app = Flask(__name__)

# ========== 全局模型加载 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# 特征顺序（与训练时保持一致）
FEATURE_ORDER = [
    'arc_voltage',
    'welding_current',
    'welding_speed',
    'wire_feed_speed',
    'gas_flow_rate',
    'torch_angle',
    'base_metal_temperature'
]

# 字段中文名
FEATURE_NAMES_CN = {
    'arc_voltage': '电弧电压',
    'welding_current': '焊接电流',
    'welding_speed': '焊接速度',
    'wire_feed_speed': '送丝速度',
    'gas_flow_rate': '气体流量',
    'torch_angle': '焊枪角度',
    'base_metal_temperature': '母材温度'
}

# 字段单位
FEATURE_UNITS = {
    'arc_voltage': 'V',
    'welding_current': 'A',
    'welding_speed': 'cm/min',
    'wire_feed_speed': 'm/min',
    'gas_flow_rate': 'L/min',
    'torch_angle': '°',
    'base_metal_temperature': '°C'
}

# 参数字段合理范围（用于输入校验）
PARAM_RANGES = {
    'arc_voltage': (0, 50),
    'welding_current': (0, 500),
    'welding_speed': (0, 100),
    'wire_feed_speed': (0, 30),
    'gas_flow_rate': (0, 50),
    'torch_angle': (0, 90),
    'base_metal_temperature': (0, 200)
}

# 懒加载模型
_model = None
_scaler = None
_feature_importance = None
_metrics = None
_feature_map = None
_orig_feature_order = None


def load_model():
    """加载分类模型和标准化器"""
    global _model, _scaler
    if _model is None:
        model_path = os.path.join(MODEL_DIR, 'classifier.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}，请先运行 algorithm/train.py")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}，请先运行 algorithm/preprocess.py")
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        print("[Server] 模型和标准化器加载完成")
    return _model, _scaler


def load_feature_map():
    """加载特征名映射（简洁名 -> 原始列名）"""
    global _feature_map, _orig_feature_order
    if _feature_map is None:
        fm_path = os.path.join(MODEL_DIR, 'feature_map.json')
        if os.path.exists(fm_path):
            with open(fm_path, 'r', encoding='utf-8') as f:
                _feature_map = json.load(f)
            # 按 FEATURE_ORDER 构造原始列名顺序
            _orig_feature_order = [_feature_map[f] for f in FEATURE_ORDER]
        else:
            _feature_map = {}
            _orig_feature_order = FEATURE_ORDER
    return _feature_map, _orig_feature_order


def load_feature_importance():
    """加载特征重要性数据"""
    global _feature_importance
    if _feature_importance is None:
        fi_path = os.path.join(MODEL_DIR, 'feature_importance.json')
        if os.path.exists(fi_path):
            with open(fi_path, 'r', encoding='utf-8') as f:
                _feature_importance = json.load(f)
        else:
            _feature_importance = []
    return _feature_importance


def load_metrics():
    """加载模型评估指标"""
    global _metrics
    if _metrics is None:
        metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                _metrics = json.load(f)
        else:
            _metrics = {}
    return _metrics


# ========== 页面路由 ==========

@app.route('/')
def index():
    """首页 - 焊接缺陷预测系统主界面"""
    return render_template('index.html')


# ========== API 接口 ==========

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    焊接缺陷预测接口
    输入：7个焊接工艺参数（JSON）
    输出：预测结果、缺陷概率、置信度
    """
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'success': False, 'error': '请求体必须为JSON格式'}), 400

        # 校验必填参数
        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({
                'success': False,
                'error': f'缺少必要参数: {", ".join(missing)}'
            }), 400

        # 提取并校验参数
        params = {}
        for field in FEATURE_ORDER:
            try:
                value = float(data[field])
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'参数 {field} ({FEATURE_NAMES_CN[field]}) 必须为数值'
                }), 400

            min_val, max_val = PARAM_RANGES[field]
            if value < min_val or value > max_val:
                return jsonify({
                    'success': False,
                    'error': f'参数 {FEATURE_NAMES_CN[field]} 超出合理范围 [{min_val}, {max_val}]'
                }), 400

            params[field] = value

        # 构造特征向量（使用原始列名，与训练时保持一致）
        _, orig_columns = load_feature_map()
        feature_df = pd.DataFrame([[params[f] for f in FEATURE_ORDER]], columns=orig_columns)

        # 加载模型并预测
        model, scaler = load_model()
        feature_scaled = pd.DataFrame(scaler.transform(feature_df), columns=orig_columns)

        prediction = int(model.predict(feature_scaled)[0])
        probabilities = model.predict_proba(feature_scaled)[0]

        # probabilities[0] = 无缺陷概率, probabilities[1] = 有缺陷概率
        defect_probability = float(probabilities[1])
        normal_probability = float(probabilities[0])
        confidence = float(max(probabilities))

        # 生成提示信息
        if prediction == 1:
            message = '存在焊接缺陷风险，建议检查工艺参数'
            status = 'warning'
        else:
            message = '焊接参数正常，预测无缺陷'
            status = 'normal'

        # 存入数据库
        record_id = insert_prediction(params, prediction, defect_probability)

        # 构造返回结果
        result = {
            'success': True,
            'record_id': record_id,
            'prediction': prediction,
            'prediction_label': '有缺陷' if prediction == 1 else '无缺陷',
            'defect_probability': round(defect_probability, 4),
            'normal_probability': round(normal_probability, 4),
            'confidence': round(confidence, 4),
            'message': message,
            'status': status,
            'params': params
        }

        return jsonify(result), 200

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
def history():
    """
    获取历史预测记录
    参数：limit（可选，默认50）
    """
    try:
        limit = request.args.get('limit', default=50, type=int)
        limit = max(1, min(limit, 500))  # 限制范围
        records = get_history(limit)
        return jsonify({
            'success': True,
            'count': len(records),
            'records': records
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/importance', methods=['GET'])
def importance():
    """获取特征重要性排序"""
    try:
        fi = load_feature_importance()
        return jsonify({
            'success': True,
            'features': fi
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def statistics():
    """获取预测统计信息"""
    try:
        stats = get_statistics()
        metrics = load_metrics()
        return jsonify({
            'success': True,
            'statistics': stats,
            'model_metrics': metrics
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查接口"""
    try:
        model, scaler = load_model()
        return jsonify({
            'success': True,
            'status': 'running',
            'model_loaded': True
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        }), 500


# ========== 启动 ==========

if __name__ == '__main__':
    # 初始化数据库
    init_db()
    # 预加载模型
    try:
        load_model()
        load_feature_map()
        load_feature_importance()
        load_metrics()
    except FileNotFoundError as e:
        print(f"[警告] {e}")
        print("请先运行数据预处理和模型训练脚本：")
        print("  python algorithm/preprocess.py")
        print("  python algorithm/train.py")

    print("\n" + "=" * 60)
    print("机器人焊接缺陷智能预测系统")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
