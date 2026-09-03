"""
数据库操作模块
功能：SQLite 数据库初始化、预测记录插入、历史记录查询
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'welding_prediction.db')

# 7个工艺参数字段
PARAM_FIELDS = [
    'arc_voltage',
    'welding_current',
    'welding_speed',
    'wire_feed_speed',
    'gas_flow_rate',
    'torch_angle',
    'base_metal_temperature'
]

# 字段中文名映射
FIELD_NAMES_CN = {
    'arc_voltage': '电弧电压',
    'welding_current': '焊接电流',
    'welding_speed': '焊接速度',
    'wire_feed_speed': '送丝速度',
    'gas_flow_rate': '气体流量',
    'torch_angle': '焊枪角度',
    'base_metal_temperature': '母材温度'
}


def get_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库，创建预测记录表"""
    conn = get_connection()
    cursor = conn.cursor()

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        arc_voltage REAL NOT NULL,
        welding_current REAL NOT NULL,
        welding_speed REAL NOT NULL,
        wire_feed_speed REAL NOT NULL,
        gas_flow_rate REAL NOT NULL,
        torch_angle REAL NOT NULL,
        base_metal_temperature REAL NOT NULL,
        prediction INTEGER NOT NULL,
        defect_probability REAL NOT NULL,
        batch_id TEXT
    )
    """
    cursor.execute(create_table_sql)

    # 迁移：如果表已存在但没有batch_id列，添加它
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'batch_id' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN batch_id TEXT")
        print("[DB] 已添加 batch_id 字段")

    conn.commit()
    conn.close()
    print(f"[DB] 数据库初始化完成: {DB_PATH}")


def insert_prediction(params, prediction, defect_probability, batch_id=None):
    """
    插入一条预测记录

    参数:
        params: dict, 包含7个工艺参数
        prediction: int, 预测结果 (0=无缺陷, 1=有缺陷)
        defect_probability: float, 缺陷概率
        batch_id: str, 批次ID（批量预测时使用，单条预测为None）

    返回:
        int: 新插入记录的ID
    """
    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    insert_sql = """
    INSERT INTO predictions (
        timestamp, arc_voltage, welding_current, welding_speed,
        wire_feed_speed, gas_flow_rate, torch_angle,
        base_metal_temperature, prediction, defect_probability, batch_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor.execute(insert_sql, (
        timestamp,
        params['arc_voltage'],
        params['welding_current'],
        params['welding_speed'],
        params['wire_feed_speed'],
        params['gas_flow_rate'],
        params['torch_angle'],
        params['base_metal_temperature'],
        prediction,
        defect_probability,
        batch_id
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_history(limit=200):
    """
    获取历史预测记录

    参数:
        limit: int, 返回记录数量上限

    返回:
        list[dict]: 历史记录列表，按时间倒序
    """
    conn = get_connection()
    cursor = conn.cursor()

    query_sql = """
    SELECT id, timestamp, arc_voltage, welding_current, welding_speed,
           wire_feed_speed, gas_flow_rate, torch_angle,
           base_metal_temperature, prediction, defect_probability, batch_id
    FROM predictions
    ORDER BY id DESC
    LIMIT ?
    """

    cursor.execute(query_sql, (limit,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            'id': row['id'],
            'timestamp': row['timestamp'],
            'arc_voltage': row['arc_voltage'],
            'welding_current': row['welding_current'],
            'welding_speed': row['welding_speed'],
            'wire_feed_speed': row['wire_feed_speed'],
            'gas_flow_rate': row['gas_flow_rate'],
            'torch_angle': row['torch_angle'],
            'base_metal_temperature': row['base_metal_temperature'],
            'prediction': row['prediction'],
            'defect_probability': round(row['defect_probability'], 4),
            'batch_id': row['batch_id']
        })

    return results


def get_history_grouped(limit=200):
    """
    获取按批次分组的历史记录

    返回:
        list[dict]: 分组后的记录列表，每个元素是一个批次或单条记录
        批次格式: {type:'batch', batch_id, timestamp, count, high_risk, medium_risk, low_risk, records:[...]}
        单条格式: {type:'single', ...记录字段}
    """
    records = get_history(limit)

    # 按批次分组
    batches = {}
    singles = []
    for r in records:
        if r['batch_id']:
            if r['batch_id'] not in batches:
                batches[r['batch_id']] = []
            batches[r['batch_id']].append(r)
        else:
            singles.append(r)

    # 构建结果列表
    result = []

    # 单条记录直接加入
    for r in singles:
        r['type'] = 'single'
        result.append(r)

    # 批次记录分组加入
    for batch_id, batch_records in batches.items():
        high = sum(1 for r in batch_records if r['defect_probability'] >= 0.6)
        medium = sum(1 for r in batch_records if 0.3 <= r['defect_probability'] < 0.6)
        low = sum(1 for r in batch_records if r['defect_probability'] < 0.3)
        result.append({
            'type': 'batch',
            'batch_id': batch_id,
            'timestamp': batch_records[0]['timestamp'],
            'count': len(batch_records),
            'high_risk': high,
            'medium_risk': medium,
            'low_risk': low,
            'records': sorted(batch_records, key=lambda x: x['id'])
        })

    # 按时间倒序排序
    result.sort(key=lambda x: x['timestamp'], reverse=True)
    return result


def get_all_history():
    """
    获取全部历史预测记录（用于导出）

    返回:
        list[dict]: 全部历史记录列表，按时间倒序
    """
    conn = get_connection()
    cursor = conn.cursor()

    query_sql = """
    SELECT id, timestamp, arc_voltage, welding_current, welding_speed,
           wire_feed_speed, gas_flow_rate, torch_angle,
           base_metal_temperature, prediction, defect_probability, batch_id
    FROM predictions
    ORDER BY id DESC
    """
    cursor.execute(query_sql)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            'id': row['id'],
            'timestamp': row['timestamp'],
            'arc_voltage': row['arc_voltage'],
            'welding_current': row['welding_current'],
            'welding_speed': row['welding_speed'],
            'wire_feed_speed': row['wire_feed_speed'],
            'gas_flow_rate': row['gas_flow_rate'],
            'torch_angle': row['torch_angle'],
            'base_metal_temperature': row['base_metal_temperature'],
            'prediction': row['prediction'],
            'defect_probability': round(row['defect_probability'], 4),
            'batch_id': row['batch_id']
        })
    return results


def clear_history():
    """
    清空所有预测记录

    返回:
        int: 删除的记录数
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def get_statistics():
    """
    获取预测统计信息

    返回:
        dict: 包含总预测数、缺陷数、正常数、缺陷率
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as defect_count FROM predictions WHERE prediction = 1")
    defect_count = cursor.fetchone()['defect_count']

    conn.close()

    normal_count = total - defect_count
    defect_rate = round(defect_count / total * 100, 2) if total > 0 else 0

    return {
        'total': total,
        'defect_count': defect_count,
        'normal_count': normal_count,
        'defect_rate': defect_rate
    }


if __name__ == '__main__':
    init_db()
    print("数据库初始化测试完成")
