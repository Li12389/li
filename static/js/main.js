/**
 * 机器人焊接缺陷智能预测系统 - 前端交互逻辑
 */

// ========== 全局变量 ==========
let importanceChart = null;
let distributionChart = null;

// 特征中文名映射
const FEATURE_NAMES_CN = {
    arc_voltage: '电弧电压',
    welding_current: '焊接电流',
    welding_speed: '焊接速度',
    wire_feed_speed: '送丝速度',
    gas_flow_rate: '气体流量',
    torch_angle: '焊枪角度',
    base_metal_temperature: '母材温度'
};

// 快捷填充示例数据
const QUICK_DATA = {
    normal: {
        arc_voltage: 22.5,
        welding_current: 180.0,
        welding_speed: 30.0,
        wire_feed_speed: 8.0,
        gas_flow_rate: 15.0,
        torch_angle: 25.0,
        base_metal_temperature: 50.0
    },
    defect: {
        arc_voltage: 18.5,
        welding_current: 290.0,
        welding_speed: 42.0,
        wire_feed_speed: 14.5,
        gas_flow_rate: 22.0,
        torch_angle: 23.0,
        base_metal_temperature: 36.0
    }
};

// ========== 工具函数 ==========

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => {
        toast.className = 'toast';
    }, 3000);
}

// ========== 预测功能 ==========

async function handlePredict(event) {
    event.preventDefault();

    const btn = document.getElementById('btnPredict');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');

    // 收集参数
    const params = {};
    const fields = ['arc_voltage', 'welding_current', 'welding_speed', 'wire_feed_speed',
                    'gas_flow_rate', 'torch_angle', 'base_metal_temperature'];

    for (const field of fields) {
        const input = document.getElementById(field);
        const value = input.value.trim();
        if (value === '') {
            showToast(`请填写${FEATURE_NAMES_CN[field]}`, 'error');
            input.focus();
            return;
        }
        params[field] = parseFloat(value);
    }

    // 显示加载状态
    btn.disabled = true;
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        const result = await response.json();

        if (result.success) {
            displayResult(result);
            showToast('预测完成', 'success');
            // 刷新历史记录和统计
            loadHistory();
            loadStatistics();
            loadDistributionChart();
        } else {
            showToast(result.error || '预测失败', 'error');
        }
    } catch (error) {
        showToast('网络错误，请检查服务是否启动', 'error');
        console.error('预测请求失败:', error);
    } finally {
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
}

function displayResult(result) {
    document.getElementById('resultPlaceholder').style.display = 'none';
    document.getElementById('resultContent').style.display = 'block';

    const badge = document.getElementById('statusBadge');
    const message = document.getElementById('resultMessage');

    if (result.prediction === 1) {
        badge.textContent = '存在缺陷';
        badge.className = 'status-badge warning';
        message.className = 'result-message warning';
    } else {
        badge.textContent = '焊接正常';
        badge.className = 'status-badge normal';
        message.className = 'result-message normal';
    }
    message.textContent = result.message;

    // 概率进度条
    const defectProb = (result.defect_probability * 100).toFixed(1);
    const normalProb = (result.normal_probability * 100).toFixed(1);

    document.getElementById('probBar').style.width = defectProb + '%';
    document.getElementById('probValue').textContent = defectProb + '%';
    document.getElementById('normalBar').style.width = normalProb + '%';
    document.getElementById('normalValue').textContent = normalProb + '%';
    document.getElementById('confValue').textContent = (result.confidence * 100).toFixed(1) + '%';
}

// ========== 历史记录 ==========

async function loadHistory() {
    try {
        const response = await fetch('/api/history?limit=50');
        const result = await response.json();

        const tbody = document.getElementById('historyBody');

        if (!result.success || result.records.length === 0) {
            tbody.innerHTML = '<tr><td colspan="11" class="empty-row">暂无预测记录</td></tr>';
            return;
        }

        tbody.innerHTML = result.records.map((record, index) => `
            <tr>
                <td>${index + 1}</td>
                <td>${record.timestamp}</td>
                <td>${record.arc_voltage.toFixed(2)}</td>
                <td>${record.welding_current.toFixed(2)}</td>
                <td>${record.welding_speed.toFixed(2)}</td>
                <td>${record.wire_feed_speed.toFixed(2)}</td>
                <td>${record.gas_flow_rate.toFixed(2)}</td>
                <td>${record.torch_angle.toFixed(2)}</td>
                <td>${record.base_metal_temperature.toFixed(2)}</td>
                <td><span class="tag-result ${record.prediction === 1 ? 'warning' : 'normal'}">
                    ${record.prediction === 1 ? '有缺陷' : '正常'}
                </span></td>
                <td>${(record.defect_probability * 100).toFixed(1)}%</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('加载历史记录失败:', error);
    }
}

// ========== 统计信息 ==========

async function loadStatistics() {
    try {
        const response = await fetch('/api/statistics');
        const result = await response.json();

        if (result.success) {
            const stats = result.statistics;
            document.getElementById('statTotal').textContent = stats.total;
            document.getElementById('statNormal').textContent = stats.normal_count;
            document.getElementById('statDefect').textContent = stats.defect_count;
            document.getElementById('statRate').textContent = stats.defect_rate + '%';

            if (result.model_metrics && result.model_metrics.accuracy) {
                document.getElementById('statAccuracy').textContent =
                    (result.model_metrics.accuracy * 100).toFixed(1) + '%';
            }
        }
    } catch (error) {
        console.error('加载统计信息失败:', error);
    }
}

// ========== ECharts 图表 ==========

async function loadImportanceChart() {
    try {
        const response = await fetch('/api/importance');
        const result = await response.json();

        if (!result.success || result.features.length === 0) return;

        const features = result.features;
        const names = features.map(f => f.feature_name_cn);
        const values = features.map(f => f.importance);

        if (!importanceChart) {
            importanceChart = echarts.init(document.getElementById('importanceChart'));
        }

        const option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow' },
                formatter: function(params) {
                    return `${params[0].name}<br/>重要性: ${(params[0].value * 100).toFixed(2)}%`;
                }
            },
            grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
            xAxis: {
                type: 'value',
                axisLabel: {
                    formatter: function(value) { return (value * 100).toFixed(0) + '%'; }
                }
            },
            yAxis: {
                type: 'category',
                data: names.reverse(),
                axisLabel: { fontSize: 12 }
            },
            series: [{
                type: 'bar',
                data: values.reverse(),
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                        { offset: 0, color: '#1890ff' },
                        { offset: 1, color: '#096dd9' }
                    ]),
                    borderRadius: [0, 4, 4, 0]
                },
                label: {
                    show: true,
                    position: 'right',
                    formatter: function(params) {
                        return (params.value * 100).toFixed(1) + '%';
                    },
                    fontSize: 11
                }
            }]
        };

        importanceChart.setOption(option);
    } catch (error) {
        console.error('加载特征重要性图表失败:', error);
    }
}

async function loadDistributionChart() {
    try {
        const response = await fetch('/api/history?limit=200');
        const result = await response.json();

        if (!result.success) return;

        const records = result.records;
        const normalCount = records.filter(r => r.prediction === 0).length;
        const defectCount = records.filter(r => r.prediction === 1).length;

        if (!distributionChart) {
            distributionChart = echarts.init(document.getElementById('distributionChart'));
        }

        const option = {
            tooltip: {
                trigger: 'item',
                formatter: '{b}: {c}次 ({d}%)'
            },
            legend: {
                bottom: 10,
                data: ['正常', '缺陷预警']
            },
            series: [{
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['50%', '45%'],
                avoidLabelOverlap: false,
                itemStyle: {
                    borderRadius: 6,
                    borderColor: '#fff',
                    borderWidth: 2
                },
                label: {
                    show: true,
                    formatter: '{b}\n{c}次'
                },
                data: [
                    { value: normalCount, name: '正常', itemStyle: { color: '#52c41a' } },
                    { value: defectCount, name: '缺陷预警', itemStyle: { color: '#ff4d4f' } }
                ]
            }]
        };

        distributionChart.setOption(option);
    } catch (error) {
        console.error('加载分布图表失败:', error);
    }
}

// ========== 快捷填充与重置 ==========

function fillQuickData(type) {
    const data = QUICK_DATA[type];
    if (!data) return;
    for (const [field, value] of Object.entries(data)) {
        document.getElementById(field).value = value;
    }
    showToast(type === 'normal' ? '已填充正常参数示例' : '已填充异常参数示例', 'info');
}

function resetForm() {
    document.getElementById('predictForm').reset();
    document.getElementById('resultPlaceholder').style.display = 'block';
    document.getElementById('resultContent').style.display = 'none';
    showToast('参数已重置', 'info');
}

// ========== 健康检查 ==========

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const result = await response.json();
        const statusDot = document.getElementById('serverStatus');
        const statusText = document.getElementById('serverStatusText');

        if (result.success) {
            statusDot.style.background = '#52c41a';
            statusDot.style.boxShadow = '0 0 8px #52c41a';
            statusText.textContent = '系统运行中';
        } else {
            statusDot.style.background = '#ff4d4f';
            statusDot.style.boxShadow = '0 0 8px #ff4d4f';
            statusText.textContent = '服务异常';
        }
    } catch (error) {
        document.getElementById('serverStatus').style.background = '#ff4d4f';
        document.getElementById('serverStatusText').textContent = '服务未连接';
    }
}

// ========== 窗口大小变化重绘图表 ==========

window.addEventListener('resize', function() {
    if (importanceChart) importanceChart.resize();
    if (distributionChart) distributionChart.resize();
});

// ========== 初始化 ==========

document.addEventListener('DOMContentLoaded', function() {
    // 绑定表单提交
    document.getElementById('predictForm').addEventListener('submit', handlePredict);

    // 绑定重置按钮
    document.getElementById('btnReset').addEventListener('click', resetForm);

    // 绑定刷新历史按钮
    document.getElementById('btnRefreshHistory').addEventListener('click', loadHistory);

    // 绑定快捷填充按钮
    document.querySelectorAll('.btn-quick').forEach(btn => {
        btn.addEventListener('click', function() {
            fillQuickData(this.dataset.type);
        });
    });

    // 初始化加载
    checkHealth();
    loadStatistics();
    loadHistory();
    loadImportanceChart();
    loadDistributionChart();

    // 定时刷新统计
    setInterval(checkHealth, 30000);
});
