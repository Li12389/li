"""测试参数优化功能"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import optimize_params, load_model, load_feature_map, load_feature_importance

# 预加载
load_model()
load_feature_map()
load_feature_importance()

# 测试用异常参数（会预测为有缺陷）
test_params = {
    'arc_voltage': 18.5,
    'welding_current': 290.0,
    'welding_speed': 42.0,
    'wire_feed_speed': 14.5,
    'gas_flow_rate': 22.0,
    'torch_angle': 23.0,
    'base_metal_temperature': 36.0
}

print("=" * 60)
print("参数优化功能测试")
print("=" * 60)

result = optimize_params(test_params)

print(f"\n原始缺陷概率: {result['original_probability']*100:.1f}%")
print(f"优化后缺陷概率: {result['optimized_probability']*100:.1f}%")
print(f"概率降低幅度: {result['improvement_percent']:.1f}%")
print(f"优化后预测结果: {result['prediction_label_after']}")

print(f"\n建议调整参数 ({len(result['adjustments'])}项):")
print("-" * 60)
for adj in result['adjustments']:
    print(f"  {adj['name_cn']:8s}: {adj['original']:>7.2f} → {adj['suggested']:>7.2f}  "
          f"({adj['direction']} {abs(adj['delta']):.2f} {adj['unit']})")

print("-" * 60)
print("\n优化后完整参数:")
for k, v in result['optimized_params'].items():
    print(f"  {k}: {v}")

print("\n测试完成！")
