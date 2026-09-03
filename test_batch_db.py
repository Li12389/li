import sys
import os
import io
sys.path.insert(0, '.')
from app import app
from database.db import get_statistics, get_history

with app.test_client() as client:
    # 先查当前记录数
    stats_before = get_statistics()
    print(f"预测前: 总记录数={stats_before['total']}")

    # 批量预测3条
    csv_data = "电弧电压,焊接电流,焊接速度,送丝速度,气体流量,焊枪角度,母材温度\n"
    csv_data += "22.5,180,30,8,15,25,50\n"
    csv_data += "18.5,290,42,14.5,22,23,36\n"
    csv_data += "25,200,28,9,18,30,60\n"
    data = {'file': (io.BytesIO(csv_data.encode('utf-8-sig')), 'test.csv')}
    resp = client.post('/api/batch-predict', data=data, content_type='multipart/form-data')
    result = resp.get_json()
    print(f"批量预测: 成功={result.get('success')}, 总数={result.get('total')}")

    # 再查记录数
    stats_after = get_statistics()
    print(f"预测后: 总记录数={stats_after['total']}")
    print(f"新增记录: {stats_after['total'] - stats_before['total']} 条")

    # 查最新3条历史
    history = get_history(3)
    print("\n最新3条记录:")
    for h in history:
        print(f"  ID={h['id']} 电压={h['arc_voltage']} 缺陷概率={h['defect_probability']:.2f} 预测={h['prediction']}")

print("\n数据库更新测试完成")
