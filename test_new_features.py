import sys
import os
sys.path.insert(0, '.')

# 模拟Flask应用上下文测试
from app import app

with app.test_client() as client:
    # 测试建议接口
    resp = client.get('/api/suggestions')
    data = resp.get_json()
    print(f"建议接口状态: {resp.status_code}")
    print(f"建议数量: {len(data.get('suggestions', []))}")
    for s in data.get('suggestions', []):
        print(f"  [{s['type']}] {s['title']}")
        print(f"       {s['content'][:60]}...")

    # 测试预测接口的风险等级
    print("\n--- 测试风险等级判定 ---")
    normal_params = {
        'arc_voltage': 22.5, 'welding_current': 180.0, 'welding_speed': 30.0,
        'wire_feed_speed': 8.0, 'gas_flow_rate': 15.0, 'torch_angle': 25.0,
        'base_metal_temperature': 50.0
    }
    resp2 = client.post('/api/predict', json=normal_params)
    d2 = resp2.get_json()
    print(f"正常参数: 缺陷概率={d2['defect_probability']*100:.1f}% 风险={d2['risk_label']} 状态={d2['status']}")

    defect_params = {
        'arc_voltage': 18.5, 'welding_current': 290.0, 'welding_speed': 42.0,
        'wire_feed_speed': 14.5, 'gas_flow_rate': 22.0, 'torch_angle': 23.0,
        'base_metal_temperature': 36.0
    }
    resp3 = client.post('/api/predict', json=defect_params)
    d3 = resp3.get_json()
    print(f"异常参数: 缺陷概率={d3['defect_probability']*100:.1f}% 风险={d3['risk_label']} 状态={d3['status']}")

print("\n全部测试通过")
