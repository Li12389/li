"""
Flask 后端服务
机器人焊接缺陷智能预测系统
功能：提供焊接缺陷预测API、历史记录查询、特征重要性查询、统计信息
"""

import os
import sys
import json
import csv
import io
import uuid
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

# 将项目根目录加入路径，确保能导入 database 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db import init_db, insert_prediction, get_history, get_history_grouped, get_statistics, get_all_history, clear_history

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

        # 三级风险判定（基于缺陷概率，而非默认50%阈值）
        if defect_probability < 0.3:
            risk_level = 'low'
            risk_label = '低风险'
            message = '当前参数组合缺陷风险较低，可正常施焊'
            status = 'normal'
        elif defect_probability < 0.6:
            risk_level = 'medium'
            risk_label = '中风险'
            message = '当前参数存在一定缺陷风险，建议关注关键参数'
            status = 'warning'
        else:
            risk_level = 'high'
            risk_label = '高风险'
            message = '当前参数缺陷风险较高，建议调整工艺参数'
            status = 'danger'

        # 存入数据库
        record_id = insert_prediction(params, prediction, defect_probability)

        # 构造返回结果
        result = {
            'success': True,
            'record_id': record_id,
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_label': risk_label,
            'prediction_label': risk_label,
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


def predict_single(params_dict):
    """
    内部预测函数，返回缺陷概率
    params_dict: 参数字典 {feature_name: value}
    返回: defect_probability (0-1)
    """
    model, scaler = load_model()
    _, orig_columns = load_feature_map()
    feature_df = pd.DataFrame([[params_dict[f] for f in FEATURE_ORDER]], columns=orig_columns)
    feature_scaled = pd.DataFrame(scaler.transform(feature_df), columns=orig_columns)
    probabilities = model.predict_proba(feature_scaled)[0]
    return float(probabilities[1])


def optimize_params(current_params):
    """
    参数优化：基于当前参数，搜索使缺陷概率最低的参数组合
    使用坐标下降法，逐个参数调整，迭代2轮
    返回: {
        'optimized_params': 优化后的参数字典,
        'original_probability': 原始缺陷概率,
        'optimized_probability': 优化后缺陷概率,
        'improvement': 概率降低幅度,
        'adjustments': [{'feature': 'arc_voltage', 'name_cn': '电弧电压', 'unit': 'V',
                         'original': 20.0, 'suggested': 22.0, 'delta': 2.0, 'direction': '增大'}]
    }
    """
    original_prob = predict_single(current_params)
    best_params = dict(current_params)
    best_prob = original_prob

    # 每个参数的搜索步长（按范围的5%取步长，至少取5个候选点）
    def get_candidates(feature, current_val):
        min_val, max_val = PARAM_RANGES[feature]
        step = (max_val - min_val) * 0.05
        candidates = set()
        candidates.add(current_val)
        # 向上下各探索4个步长
        for k in range(1, 5):
            v_up = current_val + step * k
            v_down = current_val - step * k
            if min_val <= v_up <= max_val:
                candidates.add(round(v_up, 2))
            if min_val <= v_down <= max_val:
                candidates.add(round(v_down, 2))
        return sorted(candidates)

    # 坐标下降：迭代2轮，每轮按特征重要性从高到低逐个优化
    fi = load_feature_importance()
    # 按重要性排序的特征名
    sorted_features = [item['feature'] for item in fi] if fi else FEATURE_ORDER
    # 确保所有特征都在列表中
    for f in FEATURE_ORDER:
        if f not in sorted_features:
            sorted_features.append(f)

    for iteration in range(2):
        for feature in sorted_features:
            current_val = best_params[feature]
            candidates = get_candidates(feature, current_val)
            local_best_val = current_val
            local_best_prob = best_prob
            for cand in candidates:
                test_params = dict(best_params)
                test_params[feature] = cand
                prob = predict_single(test_params)
                if prob < local_best_prob:
                    local_best_prob = prob
                    local_best_val = cand
            best_params[feature] = local_best_val
            best_prob = local_best_prob

    # 生成调整建议
    adjustments = []
    for feature in FEATURE_ORDER:
        orig_val = current_params[feature]
        opt_val = best_params[feature]
        delta = round(opt_val - orig_val, 2)
        if abs(delta) > 0.01:
            direction = '增大' if delta > 0 else '减小'
            adjustments.append({
                'feature': feature,
                'name_cn': FEATURE_NAMES_CN[feature],
                'unit': FEATURE_UNITS[feature],
                'original': round(orig_val, 2),
                'suggested': round(opt_val, 2),
                'delta': delta,
                'direction': direction
            })

    # 按调整幅度绝对值排序，影响大的排前面
    adjustments.sort(key=lambda x: abs(x['delta']), reverse=True)

    return {
        'optimized_params': {k: round(v, 2) for k, v in best_params.items()},
        'original_probability': round(original_prob, 4),
        'optimized_probability': round(best_prob, 4),
        'improvement': round(original_prob - best_prob, 4),
        'improvement_percent': round((original_prob - best_prob) / original_prob * 100, 1) if original_prob > 0 else 0,
        'adjustments': adjustments,
        'prediction_after': 1 if best_prob >= 0.5 else 0,
        'prediction_label_after': '有缺陷' if best_prob >= 0.5 else '无缺陷'
    }


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """
    参数优化建议接口
    输入：当前7个工艺参数
    输出：优化后的参数建议、调整明细、预期缺陷概率
    """
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'success': False, 'error': '请求体必须为JSON格式'}), 400

        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({
                'success': False,
                'error': f'缺少必要参数: {", ".join(missing)}'
            }), 400

        params = {}
        for field in FEATURE_ORDER:
            try:
                value = float(data[field])
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'参数 {field} 必须为数值'
                }), 400
            min_val, max_val = PARAM_RANGES[field]
            if value < min_val or value > max_val:
                return jsonify({
                    'success': False,
                    'error': f'参数 {FEATURE_NAMES_CN[field]} 超出合理范围 [{min_val}, {max_val}]'
                }), 400
            params[field] = value

        result = optimize_params(params)
        result['success'] = True
        return jsonify(result), 200

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    批量预测接口
    接收CSV文件，包含7项工艺参数，批量返回预测结果
    CSV列名支持英文(arc_voltage等)或中文(电弧电压等)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未找到上传文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': '仅支持CSV格式文件'}), 400

        # 读取CSV内容
        content = file.read().decode('utf-8-sig')
        df = pd.read_csv(io.StringIO(content))

        # 列名映射（支持多种命名方式）
        col_map = {
            # 简洁英文名
            'arc_voltage': 'arc_voltage', 'welding_current': 'welding_current',
            'welding_speed': 'welding_speed', 'wire_feed_speed': 'wire_feed_speed',
            'gas_flow_rate': 'gas_flow_rate', 'torch_angle': 'torch_angle',
            'base_metal_temperature': 'base_metal_temperature',
            # 原始带单位英文名
            'arc voltage (v)': 'arc_voltage', 'weld current (a)': 'welding_current',
            'weld speed (cm/min)': 'welding_speed', 'wire feed speed (m/min)': 'wire_feed_speed',
            'gas flow rate (l/min)': 'gas_flow_rate', 'torch angle (°)': 'torch_angle',
            'base metal temp (°c)': 'base_metal_temperature',
            # 常见英文变体
            'voltage': 'arc_voltage', 'current': 'welding_current',
            'weld speed': 'welding_speed', 'wire feed': 'wire_feed_speed',
            'gas flow': 'gas_flow_rate', 'angle': 'torch_angle',
            'base temp': 'base_metal_temperature', 'temperature': 'base_metal_temperature',
            'speed': 'welding_speed', 'temp': 'base_metal_temperature',
            'wirefeed': 'wire_feed_speed', 'gasflow': 'gas_flow_rate',
            'arc voltage': 'arc_voltage', 'weld current': 'welding_current',
            'wire feed speed': 'wire_feed_speed', 'gas flow rate': 'gas_flow_rate',
            'torch angle': 'torch_angle', 'base metal temp': 'base_metal_temperature',
            'base metal temperature': 'base_metal_temperature',
            'weldspeed': 'welding_speed', 'wirefeedspeed': 'wire_feed_speed',
            'gasflowrate': 'gas_flow_rate', 'torchangle': 'torch_angle',
            'basemetaltemp': 'base_metal_temperature',
            'arcvoltage': 'arc_voltage', 'weldcurrent': 'welding_current',
            # 中文名
            '电弧电压': 'arc_voltage', '焊接电流': 'welding_current',
            '焊接速度': 'welding_speed', '送丝速度': 'wire_feed_speed',
            '气体流量': 'gas_flow_rate', '焊枪角度': 'torch_angle',
            '母材温度': 'base_metal_temperature',
        }

        # 规范化列名后匹配（转小写、去空格、去特殊字符）
        def normalize_col(name):
            return str(name).strip().lower().replace(' ', '').replace('_', '').replace('(', '').replace(')', '').replace('/', '')

        # 构建规范化映射
        norm_map = {normalize_col(k): v for k, v in col_map.items()}

        # 重命名列
        rename_dict = {}
        for col in df.columns:
            norm = normalize_col(col)
            if norm in norm_map:
                rename_dict[col] = norm_map[norm]
            elif col.strip() in col_map:
                rename_dict[col] = col_map[col.strip()]
        df = df.rename(columns=rename_dict)

        # 检查必要列
        missing = [f for f in FEATURE_ORDER if f not in df.columns]
        if missing:
            missing_cn = [FEATURE_NAMES_CN[f] for f in missing]
            return jsonify({
                'success': False,
                'error': f'CSV缺少必要列: {", ".join(missing_cn)}'
            }), 400

        # 限制最大行数
        if len(df) > 500:
            return jsonify({'success': False, 'error': '单次最多预测500条数据'}), 400
        if len(df) == 0:
            return jsonify({'success': False, 'error': 'CSV文件为空'}), 400

        # 加载模型
        model, scaler = load_model()
        _, orig_columns = load_feature_map()

        # 提取特征并预测
        feature_df = df[FEATURE_ORDER].copy()
        # 转换为数值，非数值设为NaN
        for col in FEATURE_ORDER:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

        # 检查NaN
        nan_rows = feature_df[feature_df.isna().any(axis=1)].index.tolist()
        if nan_rows:
            return jsonify({
                'success': False,
                'error': f'第 {[r+2 for r in nan_rows]} 行存在无效数值，请检查'
            }), 400

        # 范围校验
        for field in FEATURE_ORDER:
            min_val, max_val = PARAM_RANGES[field]
            out_of_range = feature_df[(feature_df[field] < min_val) | (feature_df[field] > max_val)]
            if len(out_of_range) > 0:
                row_nums = [r + 2 for r in out_of_range.index.tolist()]
                return jsonify({
                    'success': False,
                    'error': f'{FEATURE_NAMES_CN[field]} 超出范围 [{min_val}, {max_val}]，行号: {row_nums[:5]}'
                }), 400

        # 批量预测（使用原始列名，与训练时保持一致）
        _, orig_columns = load_feature_map()
        feature_named = pd.DataFrame(feature_df.values, columns=orig_columns)
        feature_scaled = pd.DataFrame(scaler.transform(feature_named), columns=orig_columns)
        predictions = model.predict(feature_scaled)
        probabilities = model.predict_proba(feature_scaled)

        # 生成批次ID
        batch_id = 'BATCH_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]

        # 构造结果
        results = []
        for i in range(len(df)):
            defect_prob = float(probabilities[i][1])
            pred = int(predictions[i])
            if defect_prob < 0.3:
                risk = '低风险'
            elif defect_prob < 0.6:
                risk = '中风险'
            else:
                risk = '高风险'

            params_dict = {f: round(float(feature_df.iloc[i][f]), 2) for f in FEATURE_ORDER}
            row = {
                'row': i + 1,
                'params': params_dict,
                'defect_probability': round(defect_prob, 4),
                'risk_level': risk,
                'prediction': pred
            }
            results.append(row)

            # 写入数据库
            try:
                insert_prediction(params_dict, pred, defect_prob, batch_id=batch_id)
            except Exception as db_err:
                print(f"[Server] 批量预测写入数据库失败(第{i+1}条): {db_err}")

        # 统计
        high_risk = sum(1 for r in results if r['risk_level'] == '高风险')
        medium_risk = sum(1 for r in results if r['risk_level'] == '中风险')
        low_risk = sum(1 for r in results if r['risk_level'] == '低风险')

        return jsonify({
            'success': True,
            'total': len(results),
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'results': results
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': f'批量预测失败: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
def history():
    """
    获取历史预测记录（按批次分组）
    参数：limit（可选，默认200）
    """
    try:
        limit = request.args.get('limit', default=200, type=int)
        limit = max(1, min(limit, 1000))  # 限制范围
        records = get_history_grouped(limit)
        return jsonify({
            'success': True,
            'count': len(records),
            'records': records
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export', methods=['GET'])
def export_history():
    """导出历史预测记录为CSV文件"""
    try:
        records = get_all_history()
        if not records:
            return jsonify({'success': False, 'error': '暂无预测记录可导出'}), 400

        # 生成CSV内容
        output = io.StringIO()
        writer = csv.writer(output)
        # 表头
        writer.writerow([
            '序号', '预测时间', '电弧电压(V)', '焊接电流(A)', '焊接速度(cm/min)',
            '送丝速度(m/min)', '气体流量(L/min)', '焊枪角度(°)', '母材温度(°C)',
            '预测结果', '缺陷概率'
        ])
        # 数据行
        for idx, r in enumerate(records, 1):
            writer.writerow([
                idx,
                r['timestamp'],
                r['arc_voltage'],
                r['welding_current'],
                r['welding_speed'],
                r['wire_feed_speed'],
                r['gas_flow_rate'],
                r['torch_angle'],
                r['base_metal_temperature'],
                '有缺陷' if r['prediction'] == 1 else '无缺陷',
                f"{r['defect_probability']*100:.2f}%"
            ])

        csv_content = output.getvalue()
        output.close()

        # 返回CSV文件，加BOM头确保Excel中文正常
        return Response(
            '\ufeff' + csv_content,
            mimetype='text/csv; charset=utf-8',
            headers={
                'Content-Disposition': 'attachment; filename=welding_prediction_history.csv'
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'error': f'导出失败: {str(e)}'}), 500


@app.route('/api/history/clear', methods=['DELETE'])
def clear_history_api():
    """清空所有历史预测记录"""
    try:
        deleted = clear_history()
        return jsonify({
            'success': True,
            'message': f'已清空 {deleted} 条预测记录',
            'deleted_count': deleted
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


@app.route('/api/suggestions', methods=['GET'])
def suggestions():
    """
    智能工艺建议接口
    根据特征重要性和历史统计数据，生成焊接工艺改进建议
    """
    try:
        fi = load_feature_importance()
        stats = get_statistics()
        metrics = load_metrics()

        suggestions_list = []

        # 1. 基于特征重要性的参数控制建议
        if fi and len(fi) >= 3:
            top3 = fi[:3]
            top_names = '、'.join([item['feature_name_cn'] for item in top3])
            suggestions_list.append({
                'type': 'focus',
                'icon': '&#127919;',
                'title': '重点控制参数',
                'content': f'根据模型分析，对缺陷影响最大的三项参数为{top_names}，建议在实际生产中优先监控和稳定这三项参数。'
            })

            # 影响最大参数的具体建议
            top_feature = fi[0]
            top_name = top_feature['feature_name_cn']
            top_importance = top_feature['importance'] * 100
            suggestions_list.append({
                'type': 'detail',
                'icon': '&#128200;',
                'title': f'{top_name}控制建议',
                'content': f'{top_name}的特征重要性为{top_importance:.1f}%，排名第一。建议将该参数波动范围控制在±5%以内，每班次至少校验一次。'
            })

        # 2. 基于历史缺陷率的质量评估
        if stats and stats['total'] > 0:
            rate = stats['defect_rate']
            if rate < 20:
                quality = '良好'
                advice = '当前整体缺陷率较低，工艺参数稳定性较好，建议保持现有参数设置。'
            elif rate < 40:
                quality = '一般'
                advice = '当前缺陷率处于中等水平，建议排查高风险参数组合，加强过程监控。'
            else:
                quality = '待改进'
                advice = '当前缺陷率偏高，建议系统性优化工艺参数，必要时进行参数试验验证。'
            suggestions_list.append({
                'type': 'quality',
                'icon': '&#128202;',
                'title': f'整体质量评估：{quality}',
                'content': f'累计预测{stats["total"]}次，缺陷率{rate}%。{advice}'
            })

        # 3. 模型性能说明
        if metrics and metrics.get('accuracy'):
            acc = metrics['accuracy'] * 100
            recall = metrics.get('recall', 0) * 100
            suggestions_list.append({
                'type': 'model',
                'icon': '&#129302;',
                'title': '模型使用建议',
                'content': f'模型准确率{acc:.1f}%，召回率{recall:.1f}%。预测结果仅供参考，实际焊接质量仍需结合无损检测进行最终确认。'
            })

        # 4. 通用操作建议
        suggestions_list.append({
            'type': 'tip',
            'icon': '&#128161;',
            'title': '使用提示',
            'content': '当预测结果为中风险或高风险时，系统会自动生成参数优化建议，可点击"应用建议参数"快速调整。建议将优化后的参数重新预测验证。'
        })

        return jsonify({
            'success': True,
            'suggestions': suggestions_list
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
