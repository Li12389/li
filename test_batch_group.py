import sys
import os
import io
sys.path.insert(0, '.')
from app import app
from database.db import get_history_grouped

with app.test_client() as client:
    # 批量预测3条
    csv_data = "电弧电压,焊接电流,焊接速度,送丝速度,气体流量,焊枪角度,母材温度\n"
    csv_data += "22.5,180,30,8,15,25,50\n"
    csv_data += "18.5,290,42,14.5,22,23,36\n"
    csv_data += "25,200,28,9,18,30,60\n"
    data = {'file': (io.BytesIO(csv_data.encode('utf-8-sig')), 'test.csv')}
    resp = client.post('/api/batch-predict', data=data, content_type='multipart/form-data')
    result = resp.get_json()
    print(f"批量预测: 成功={result.get('success')}, 总数={result.get('total')}")

    # 查分组历史记录
    grouped = get_history_grouped(50)
    print(f"\n分组后记录数: {len(grouped)}")
    for item in grouped[:5]:
        if item['type'] == 'batch':
            print(f"  [批次] {item['timestamp']} 共{item['count']}条 "
                  f"低={item['low_risk']} 中={item['medium_risk']} 高={item['high_risk']}")
        else:
            print(f"  [单条] ID={item['id']} {item['timestamp']} 缺陷概率={item['defect_probability']:.2f}")

    # 测试API接口
    print("\n--- 测试API接口 ---")
    resp2 = client.get('/api/history?limit=10')
    r2 = resp2.get_json()
    print(f"API返回记录数: {r2['count']}")
    for item in r2['records'][:3]:
        print(f"  type={item['type']}", end=' ')
        if item['type'] == 'batch':
            print(f"批次 {item['count']}条")
        else:
            print(f"单条 ID={item['id']}")

print("\n批次分组测试完成")
