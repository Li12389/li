"""
自动化测试脚本
测试内容：数据库操作、API接口、模型预测、输入校验
运行方式：python tests/test_app.py
"""

import os
import sys
import json
import tempfile
import unittest

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 使用临时数据库进行测试
TEST_DB = os.path.join(tempfile.gettempdir(), 'test_welding.db')

# 在导入 app 前设置环境变量，使用测试数据库
os.environ['TESTING'] = 'True'

from app import app
from database import db as db_module


class TestDatabase(unittest.TestCase):
    """数据库模块测试"""

    def setUp(self):
        """每个测试前使用临时数据库"""
        self.original_db_path = db_module.DB_PATH
        db_module.DB_PATH = TEST_DB
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        db_module.init_db()

    def tearDown(self):
        """测试后清理"""
        db_module.DB_PATH = self.original_db_path
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def test_init_db(self):
        """测试数据库初始化"""
        # 再次初始化不应报错
        db_module.init_db()
        self.assertTrue(os.path.exists(TEST_DB))

    def test_insert_and_query(self):
        """测试插入记录和查询历史"""
        params = {
            'arc_voltage': 22.5,
            'welding_current': 180.0,
            'welding_speed': 30.0,
            'wire_feed_speed': 8.0,
            'gas_flow_rate': 15.0,
            'torch_angle': 25.0,
            'base_metal_temperature': 50.0
        }

        # 插入记录
        record_id = db_module.insert_prediction(params, 0, 0.15)
        self.assertIsInstance(record_id, int)
        self.assertGreater(record_id, 0)

        # 查询历史
        history = db_module.get_history(limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['arc_voltage'], 22.5)
        self.assertEqual(history[0]['prediction'], 0)
        self.assertEqual(history[0]['defect_probability'], 0.15)

    def test_statistics(self):
        """测试统计信息"""
        params = {
            'arc_voltage': 22.5, 'welding_current': 180.0, 'welding_speed': 30.0,
            'wire_feed_speed': 8.0, 'gas_flow_rate': 15.0, 'torch_angle': 25.0,
            'base_metal_temperature': 50.0
        }
        db_module.insert_prediction(params, 0, 0.1)
        db_module.insert_prediction(params, 1, 0.8)

        stats = db_module.get_statistics()
        self.assertEqual(stats['total'], 2)
        self.assertEqual(stats['normal_count'], 1)
        self.assertEqual(stats['defect_count'], 1)
        self.assertEqual(stats['defect_rate'], 50.0)

    def test_empty_statistics(self):
        """测试空数据库统计"""
        stats = db_module.get_statistics()
        self.assertEqual(stats['total'], 0)
        self.assertEqual(stats['defect_rate'], 0)


class TestAPI(unittest.TestCase):
    """API接口测试"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化：使用临时数据库"""
        cls.original_db_path = db_module.DB_PATH
        db_module.DB_PATH = TEST_DB
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        db_module.init_db()
        app.config['TESTING'] = True
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        db_module.DB_PATH = cls.original_db_path
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def test_health_check(self):
        """测试健康检查接口"""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'running')

    def test_index_page(self):
        """测试首页加载"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('机器人焊接缺陷智能预测系统', response.data.decode('utf-8'))

    def test_predict_success(self):
        """测试正常预测请求"""
        params = {
            'arc_voltage': 22.5,
            'welding_current': 180.0,
            'welding_speed': 30.0,
            'wire_feed_speed': 8.0,
            'gas_flow_rate': 15.0,
            'torch_angle': 25.0,
            'base_metal_temperature': 50.0
        }
        response = self.client.post('/api/predict',
                                    data=json.dumps(params),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn(data['prediction'], [0, 1])
        self.assertIn('defect_probability', data)
        self.assertGreaterEqual(data['defect_probability'], 0)
        self.assertLessEqual(data['defect_probability'], 1)
        self.assertIn('confidence', data)
        self.assertIn('message', data)
        self.assertIn('record_id', data)

    def test_predict_missing_params(self):
        """测试缺少参数的情况"""
        params = {'arc_voltage': 22.5}  # 只传一个参数
        response = self.client.post('/api/predict',
                                    data=json.dumps(params),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('缺少必要参数', data['error'])

    def test_predict_invalid_type(self):
        """测试参数类型错误"""
        params = {
            'arc_voltage': 'abc',  # 非数值
            'welding_current': 180.0,
            'welding_speed': 30.0,
            'wire_feed_speed': 8.0,
            'gas_flow_rate': 15.0,
            'torch_angle': 25.0,
            'base_metal_temperature': 50.0
        }
        response = self.client.post('/api/predict',
                                    data=json.dumps(params),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])

    def test_predict_out_of_range(self):
        """测试参数超出范围"""
        params = {
            'arc_voltage': 999,  # 超出范围
            'welding_current': 180.0,
            'welding_speed': 30.0,
            'wire_feed_speed': 8.0,
            'gas_flow_rate': 15.0,
            'torch_angle': 25.0,
            'base_metal_temperature': 50.0
        }
        response = self.client.post('/api/predict',
                                    data=json.dumps(params),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('超出合理范围', data['error'])

    def test_predict_no_json(self):
        """测试非JSON请求体"""
        response = self.client.post('/api/predict',
                                    data='not json',
                                    content_type='text/plain')
        self.assertEqual(response.status_code, 400)

    def test_history(self):
        """测试历史记录接口"""
        # 先插入一条预测
        params = {
            'arc_voltage': 22.5, 'welding_current': 180.0, 'welding_speed': 30.0,
            'wire_feed_speed': 8.0, 'gas_flow_rate': 15.0, 'torch_angle': 25.0,
            'base_metal_temperature': 50.0
        }
        self.client.post('/api/predict', data=json.dumps(params),
                         content_type='application/json')

        response = self.client.get('/api/history?limit=10')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertGreaterEqual(data['count'], 1)
        self.assertIsInstance(data['records'], list)

    def test_history_limit(self):
        """测试历史记录limit参数"""
        response = self.client.get('/api/history?limit=5')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertLessEqual(len(data['records']), 5)

    def test_importance(self):
        """测试特征重要性接口"""
        response = self.client.get('/api/importance')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIsInstance(data['features'], list)
        self.assertEqual(len(data['features']), 7)
        # 验证每个特征都有重要性值
        for feature in data['features']:
            self.assertIn('feature', feature)
            self.assertIn('feature_name_cn', feature)
            self.assertIn('importance', feature)
            self.assertGreater(feature['importance'], 0)

    def test_statistics(self):
        """测试统计信息接口"""
        response = self.client.get('/api/statistics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('statistics', data)
        self.assertIn('model_metrics', data)
        self.assertIn('total', data['statistics'])


class TestModel(unittest.TestCase):
    """模型相关测试"""

    def test_model_files_exist(self):
        """测试模型文件存在"""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'model', 'classifier.pkl')),
                        "分类模型文件不存在")
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'model', 'scaler.pkl')),
                        "标准化器文件不存在")

    def test_feature_map_exists(self):
        """测试特征映射文件存在"""
        path = os.path.join(PROJECT_ROOT, 'model', 'feature_map.json')
        self.assertTrue(os.path.exists(path), "特征映射文件不存在")
        with open(path, 'r', encoding='utf-8') as f:
            feature_map = json.load(f)
        self.assertEqual(len(feature_map), 7)

    def test_metrics_exist(self):
        """测试评估指标文件存在"""
        path = os.path.join(PROJECT_ROOT, 'model', 'metrics.json')
        self.assertTrue(os.path.exists(path), "评估指标文件不存在")
        with open(path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始运行自动化测试...")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestModel))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("所有测试通过！")
    else:
        print(f"测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
