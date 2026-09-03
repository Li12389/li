import sys
import os
import io
sys.path.insert(0, '.')
from app import app

with app.test_client() as client:
    # 构造测试CSV数据
    csv_data = "电弧电压,焊接电流,焊接速度,送丝速度,气体流量,焊枪角度,母材温度\n"
    csv_data += "22.5,180,30,8,15,25,50\n"
    csv_data += "18.5,290,42,14.5,22,23,36\n"
    csv_data += "25,200,28,9,18,30,60\n"

    data = {'file': (io.BytesIO(csv_data.encode('utf-8-sig')), 'test.csv')}
    resp = client.post('/api/batch-predict', data=data, content_type='multipart/form-data')
    result = resp.get_json()

    print(f"接口状态: {resp.status_code}")
    print(f"成功: {result.get('success')}")
    if result.get('success'):
        print(f"总数: {result['total']}")
        print(f"低风险: {result['low_risk']}  中风险: {result['medium_risk']}  高风险: {result['high_risk']}")
        print("\n预测结果:")
        for r in result['results']:
            p = r['params']
            print(f"  第{r['row']}条: 电压={p['arc_voltage']} 电流={p['welding_current']} "
                  f"缺陷概率={r['defect_probability']*100:.1f}% 风险={r['risk_level']}")
    else:
        print(f"错误: {result.get('error')}")

    # 测试英文列名
    print("\n--- 测试英文列名 ---")
    csv_en = "arc_voltage,welding_current,welding_speed,wire_feed_speed,gas_flow_rate,torch_angle,base_metal_temperature\n"
    csv_en += "22.5,180,30,8,15,25,50\n"
    data2 = {'file': (io.BytesIO(csv_en.encode('utf-8-sig')), 'test_en.csv')}
    resp2 = client.post('/api/batch-predict', data=data2, content_type='multipart/form-data')
    r2 = resp2.get_json()
    print(f"英文列名测试: {'成功' if r2.get('success') else '失败 - ' + str(r2.get('error'))}")

    # 测试缺少列
    print("\n--- 测试缺少列 ---")
    csv_bad = "电弧电压,焊接电流\n22.5,180\n"
    data3 = {'file': (io.BytesIO(csv_bad.encode('utf-8-sig')), 'bad.csv')}
    resp3 = client.post('/api/batch-predict', data=data3, content_type='multipart/form-data')
    r3 = resp3.get_json()
    print(f"缺少列测试: {r3.get('error', '未知错误')}")

print("\n批量预测功能测试完成")
