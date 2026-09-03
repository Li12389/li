import sys
sys.path.insert(0, '.')
from database.db import get_all_history, get_statistics

records = get_all_history()
print(f'当前记录数: {len(records)}')
if records:
    r = records[0]
    label = '有缺陷' if r['prediction'] == 1 else '无缺陷'
    print(f'最新记录: {r["timestamp"]} 结果={label}')
stats = get_statistics()
print(f'统计: 总数={stats["total"]} 缺陷={stats["defect_count"]} 缺陷率={stats["defect_rate"]}%')
print('数据库函数测试通过')
