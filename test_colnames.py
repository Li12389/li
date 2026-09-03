import sys
import os
import io
sys.path.insert(0, '.')
from app import app

with app.test_client() as client:
    # 测试原始带单位的英文列名
    print("--- 测试原始带单位英文名 ---")
    csv_data = "Arc Voltage (V),Weld Current (A),Weld Speed (cm/min),Wire Feed Speed (m/min),Gas Flow Rate (L/min),Torch Angle (°),Base Metal Temp (°C)\n"
    csv_data += "22.5,180,30,8,15,25,50\n"
    csv_data += "18.5,290,42,14.5,22,23,36\n"
    data = {'file': (io.BytesIO(csv_data.encode('utf-8-sig')), 'test_original.csv')}
    resp = client.post('/api/batch-predict', data=data, content_type='multipart/form-data')
    result = resp.get_json()
    print(f"状态: {resp.status_code}, 成功: {result.get('success')}")
    if result.get('success'):
        print(f"总数: {result['total']}, 高风险: {result['high_risk']}")
    else:
        print(f"错误: {result.get('error')}")

    # 测试简单英文名（Voltage, Current等）
    print("\n--- 测试简单英文名 ---")
    csv2 = "Voltage,Current,Speed,WireFeed,GasFlow,Angle,Temp\n22.5,180,30,8,15,25,50\n"
    data2 = {'file': (io.BytesIO(csv2.encode('utf-8-sig')), 'test_simple.csv')}
    resp2 = client.post('/api/batch-predict', data=data2, content_type='multipart/form-data')
    r2 = resp2.get_json()
    print(f"状态: {resp2.status_code}, 成功: {r2.get('success')}")
    if not r2.get('success'):
        print(f"错误: {r2.get('error')}")

    # 测试中文名
    print("\n--- 测试中文名 ---")
    csv3 = "电弧电压,焊接电流,焊接速度,送丝速度,气体流量,焊枪角度,母材温度\n22.5,180,30,8,15,25,50\n"
    data3 = {'file': (io.BytesIO(csv3.encode('utf-8-sig')), 'test_cn.csv')}
    resp3 = client.post('/api/batch-predict', data=data3, content_type='multipart/form-data')
    r3 = resp3.get_json()
    print(f"状态: {resp3.status_code}, 成功: {r3.get('success')}")

print("\n全部列名格式测试完成")
