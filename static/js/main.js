/**
 * 机器人焊接缺陷智能预测系统 - 前端交互逻辑
 */

// ========== 全局变量 ==========
let importanceChart = null;
let distributionChart = null;
let currentTab = 'predict';

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
    document.getElementById('resultContent').dataset.hasResult = 'true';
    document.getElementById('batchResultContent').style.display = 'none';

    const badge = document.getElementById('statusBadge');
    const message = document.getElementById('resultMessage');

    // 三级风险显示
    badge.textContent = result.risk_label;
    badge.className = 'status-badge ' + result.status;
    message.className = 'result-message ' + result.status;
    message.textContent = result.message;

    // 中风险及以上触发参数优化建议
    if (result.defect_probability >= 0.3) {
        requestOptimize(result.params);
    } else {
        document.getElementById('optimizeSection').style.display = 'none';
    }

    // 概率进度条
    const defectProb = (result.defect_probability * 100).toFixed(1);
    const normalProb = (result.normal_probability * 100).toFixed(1);

    document.getElementById('probBar').style.width = defectProb + '%';
    document.getElementById('probValue').textContent = defectProb + '%';
    document.getElementById('normalBar').style.width = normalProb + '%';
    document.getElementById('normalValue').textContent = normalProb + '%';
    document.getElementById('confValue').textContent = (result.confidence * 100).toFixed(1) + '%';
}

// ========== 参数优化建议 ==========

let lastOptimizeResult = null;

async function requestOptimize(params) {
    const section = document.getElementById('optimizeSection');
    const loading = document.getElementById('optimizeLoading');
    const content = document.getElementById('optimizeContent');

    section.style.display = 'block';
    loading.style.display = 'block';
    content.style.display = 'none';

    try {
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const result = await response.json();

        if (result.success) {
            lastOptimizeResult = result;
            displayOptimizeResult(result);
        } else {
            loading.style.display = 'none';
            section.style.display = 'none';
            showToast('优化建议生成失败: ' + (result.error || '未知错误'), 'error');
        }
    } catch (error) {
        loading.style.display = 'none';
        section.style.display = 'none';
        console.error('优化请求失败:', error);
    }
}

function displayOptimizeResult(result) {
    document.getElementById('optimizeLoading').style.display = 'none';
    document.getElementById('optimizeContent').style.display = 'block';

    const origProb = (result.original_probability * 100).toFixed(1);
    const newProb = (result.optimized_probability * 100).toFixed(1);
    const improve = result.improvement_percent;

    document.getElementById('optOrigProb').textContent = origProb + '%';
    document.getElementById('optNewProb').textContent = newProb + '%';
    document.getElementById('optImprove').textContent = improve + '%';

    const resultLabel = document.getElementById('optResultLabel');
    if (result.prediction_after === 0) {
        resultLabel.textContent = '调整后预测结果：无缺陷，建议采纳以上参数';
        resultLabel.style.background = '#f6ffed';
        resultLabel.style.color = '#389e0d';
    } else {
        resultLabel.textContent = '调整后仍有缺陷风险，但概率已降低，建议结合实际工艺进一步调整';
        resultLabel.style.background = '#fff7e6';
        resultLabel.style.color = '#d48806';
    }

    // 填充调整建议表格
    const tbody = document.getElementById('optAdjustBody');
    if (result.adjustments.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="padding:10px; text-align:center; color:#999;">当前参数已接近最优，无需大幅调整</td></tr>';
    } else {
        tbody.innerHTML = result.adjustments.map(adj => `
            <tr>
                <td style="padding:6px 8px; border-bottom:1px solid #f0f0f0;">${adj.name_cn} (${adj.unit})</td>
                <td style="padding:6px 8px; text-align:center; border-bottom:1px solid #f0f0f0; color:#999;">${adj.original}</td>
                <td style="padding:6px 8px; text-align:center; border-bottom:1px solid #f0f0f0; color:#1890ff; font-weight:bold;">${adj.suggested}</td>
                <td style="padding:6px 8px; text-align:center; border-bottom:1px solid #f0f0f0; color:${adj.delta > 0 ? '#52c41a' : '#ff4d4f'};">
                    ${adj.direction} ${Math.abs(adj.delta)}
                </td>
            </tr>
        `).join('');
    }
}

function applyOptimizeParams() {
    if (!lastOptimizeResult) return;
    const opt = lastOptimizeResult.optimized_params;
    for (const [field, value] of Object.entries(opt)) {
        const input = document.getElementById(field);
        if (input) input.value = value;
    }
    showToast('已应用建议参数，可重新预测验证', 'success');
}

// ========== 历史记录 ==========

async function loadHistory() {
    try {
        const response = await fetch('/api/history?limit=500');
        const result = await response.json();

        const tbody = document.getElementById('historyBody');

        if (!result.success || result.records.length === 0) {
            tbody.innerHTML = '<tr><td colspan="11" class="empty-row">暂无预测记录</td></tr>';
            return;
        }

        let html = '';
        let singleIndex = 0;

        result.records.forEach((item) => {
            if (item.type === 'single') {
                singleIndex++;
                html += renderSingleRow(item, singleIndex);
            } else if (item.type === 'batch') {
                html += renderBatchRow(item);
            }
        });

        tbody.innerHTML = html;

        // 绑定批次展开/折叠事件
        tbody.querySelectorAll('.batch-toggle').forEach(el => {
            el.addEventListener('click', function() {
                const batchId = this.dataset.batchId;
                const subRows = tbody.querySelectorAll('.batch-sub-' + batchId);
                const arrow = this.querySelector('.batch-arrow');
                const isHidden = subRows[0].style.display === 'none';
                subRows.forEach(row => {
                    row.style.display = isHidden ? 'table-row' : 'none';
                });
                arrow.textContent = isHidden ? '▼' : '▶';
            });
        });
    } catch (error) {
        console.error('加载历史记录失败:', error);
    }
}

function renderSingleRow(record, index) {
    const riskClass = record.defect_probability >= 0.6 ? 'warning' :
                      record.defect_probability >= 0.3 ? 'medium' : 'normal';
    const riskText = record.defect_probability >= 0.6 ? '高风险' :
                     record.defect_probability >= 0.3 ? '中风险' : '低风险';
    return `
        <tr>
            <td>${index}</td>
            <td>${record.timestamp}</td>
            <td>${record.arc_voltage.toFixed(2)}</td>
            <td>${record.welding_current.toFixed(2)}</td>
            <td>${record.welding_speed.toFixed(2)}</td>
            <td>${record.wire_feed_speed.toFixed(2)}</td>
            <td>${record.gas_flow_rate.toFixed(2)}</td>
            <td>${record.torch_angle.toFixed(2)}</td>
            <td>${record.base_metal_temperature.toFixed(2)}</td>
            <td><span class="tag-result ${riskClass}">${riskText}</span></td>
            <td>${(record.defect_probability * 100).toFixed(1)}%</td>
        </tr>
    `;
}

function renderBatchRow(batch) {
    // 用batch_id的后8位作为唯一标识
    const shortId = batch.batch_id.replace(/[^a-zA-Z0-9]/g, '').slice(-8);
    let rowsHtml = batch.records.map((r, i) => `
        <tr class="batch-sub-row batch-sub-${shortId}" style="display:none;">
            <td>${i + 1}</td>
            <td>${r.timestamp}</td>
            <td>${r.arc_voltage.toFixed(2)}</td>
            <td>${r.welding_current.toFixed(2)}</td>
            <td>${r.welding_speed.toFixed(2)}</td>
            <td>${r.wire_feed_speed.toFixed(2)}</td>
            <td>${r.gas_flow_rate.toFixed(2)}</td>
            <td>${r.torch_angle.toFixed(2)}</td>
            <td>${r.base_metal_temperature.toFixed(2)}</td>
            <td><span class="tag-result ${r.defect_probability >= 0.6 ? 'warning' : r.defect_probability >= 0.3 ? 'medium' : 'normal'}">
                ${r.defect_probability >= 0.6 ? '高风险' : r.defect_probability >= 0.3 ? '中风险' : '低风险'}
            </span></td>
            <td>${(r.defect_probability * 100).toFixed(1)}%</td>
        </tr>
    `).join('');

    const headerRow = `
        <tr class="batch-header-row batch-toggle" data-batch-id="${shortId}" style="cursor:pointer;">
            <td colspan="11">
                <div class="batch-header">
                    <span class="batch-arrow">▶</span>
                    <span class="batch-icon">&#128230;</span>
                    <span class="batch-title">批量预测</span>
                    <span class="batch-time">${batch.timestamp}</span>
                    <span class="batch-count">共 ${batch.count} 条</span>
                    <span class="batch-risk low">低风险 ${batch.low_risk}</span>
                    <span class="batch-risk medium">中风险 ${batch.medium_risk}</span>
                    <span class="batch-risk high">高风险 ${batch.high_risk}</span>
                </div>
            </td>
        </tr>
    `;

    return headerRow + rowsHtml;
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
        const response = await fetch('/api/history?limit=500');
        const result = await response.json();

        if (!result.success) return;

        const records = result.records;
        let normalCount = 0;
        let defectCount = 0;

        // 处理分组数据：单条直接统计，批次遍历子记录
        records.forEach(item => {
            if (item.type === 'single') {
                if (item.prediction === 0) normalCount++;
                else defectCount++;
            } else if (item.type === 'batch') {
                item.records.forEach(r => {
                    if (r.prediction === 0) normalCount++;
                    else defectCount++;
                });
            }
        });

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

// ========== 预测模式切换（单条/批量） ==========

let currentPredictMode = 'single';

function switchPredictMode(mode) {
    currentPredictMode = mode;
    document.querySelectorAll('.predict-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.mode === mode);
    });
    document.getElementById('singlePredictMode').style.display = mode === 'single' ? 'block' : 'none';
    document.getElementById('batchPredictMode').style.display = mode === 'batch' ? 'block' : 'none';

    // 切换右侧结果显示
    if (mode === 'batch') {
        document.getElementById('resultContent').style.display = 'none';
        document.getElementById('resultPlaceholder').style.display = 'none';
        if (batchResults) {
            document.getElementById('batchResultContent').style.display = 'block';
        } else {
            document.getElementById('batchResultContent').style.display = 'none';
            document.getElementById('resultPlaceholder').style.display = 'block';
            document.getElementById('resultPlaceholder').querySelector('p').textContent = '请在左侧上传CSV文件进行批量预测';
        }
    } else {
        document.getElementById('batchResultContent').style.display = 'none';
        document.getElementById('resultPlaceholder').querySelector('p').textContent = '请输入焊接工艺参数后点击"开始预测"';
        // 如果有单条预测结果就显示，否则显示占位
        if (document.getElementById('resultContent').dataset.hasResult === 'true') {
            document.getElementById('resultContent').style.display = 'block';
            document.getElementById('resultPlaceholder').style.display = 'none';
        } else {
            document.getElementById('resultContent').style.display = 'none';
            document.getElementById('resultPlaceholder').style.display = 'block';
        }
    }
}

// ========== 批量预测 ==========

let batchResults = null;

function initBatchUpload() {
    const dropZone = document.getElementById('batchDropZone');
    const fileInput = document.getElementById('batchFileInput');
    const selectBtn = document.getElementById('btnSelectFile');

    selectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleBatchFile(e.target.files[0]);
        }
    });

    // 拖拽上传
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleBatchFile(e.dataTransfer.files[0]);
        }
    });

    // 导出批量结果
    document.getElementById('btnExportBatch').addEventListener('click', exportBatchResult);

    // 下载模板
    document.getElementById('downloadTemplate').addEventListener('click', (e) => {
        e.preventDefault();
        downloadCsvTemplate();
    });
}

async function handleBatchFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('请上传CSV格式文件', 'error');
        return;
    }

    document.getElementById('batchResultContent').style.display = 'none';
    document.getElementById('batchLoading').style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/batch-predict', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        document.getElementById('batchLoading').style.display = 'none';

        if (result.success) {
            batchResults = result;
            displayBatchResult(result);
            // 刷新历史记录、统计数据和分布图
            loadHistory();
            loadStatistics();
            loadDistributionChart();
            showToast(`批量预测完成，共${result.total}条，已保存到历史记录`, 'success');
        } else {
            showToast(result.error || '批量预测失败', 'error');
        }
    } catch (error) {
        document.getElementById('batchLoading').style.display = 'none';
        showToast('网络错误', 'error');
        console.error('批量预测失败:', error);
    }
}

function displayBatchResult(result) {
    // 隐藏单条结果和占位，显示批量结果
    document.getElementById('resultContent').style.display = 'none';
    document.getElementById('resultPlaceholder').style.display = 'none';
    document.getElementById('batchResultContent').style.display = 'block';

    document.getElementById('batchTotal').textContent = result.total;
    document.getElementById('batchLow').textContent = result.low_risk;
    document.getElementById('batchMedium').textContent = result.medium_risk;
    document.getElementById('batchHigh').textContent = result.high_risk;

    const tbody = document.getElementById('batchResultBody');
    tbody.innerHTML = result.results.map(r => {
        const riskClass = r.risk_level === '低风险' ? 'risk-low' :
                          r.risk_level === '中风险' ? 'risk-medium' : 'risk-high';
        const p = r.params;
        return `
            <tr>
                <td>${r.row}</td>
                <td>${p.arc_voltage}</td>
                <td>${p.welding_current}</td>
                <td>${p.welding_speed}</td>
                <td>${p.wire_feed_speed}</td>
                <td>${p.gas_flow_rate}</td>
                <td>${p.torch_angle}</td>
                <td>${p.base_metal_temperature}</td>
                <td>${(r.defect_probability * 100).toFixed(1)}%</td>
                <td class="${riskClass}">${r.risk_level}</td>
            </tr>
        `;
    }).join('');
}

function exportBatchResult() {
    if (!batchResults) return;

    let csv = '\ufeff序号,电弧电压(V),焊接电流(A),焊接速度(cm/min),送丝速度(m/min),气体流量(L/min),焊枪角度(°),母材温度(°C),缺陷概率,风险等级\n';
    batchResults.results.forEach(r => {
        const p = r.params;
        csv += `${r.row},${p.arc_voltage},${p.welding_current},${p.welding_speed},${p.wire_feed_speed},${p.gas_flow_rate},${p.torch_angle},${p.base_metal_temperature},${(r.defect_probability*100).toFixed(2)}%,${r.risk_level}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = '批量预测结果.csv';
    link.click();
    showToast('预测结果已导出', 'success');
}

function downloadCsvTemplate() {
    const csv = '\ufeff电弧电压,焊接电流,焊接速度,送丝速度,气体流量,焊枪角度,母材温度\n22.5,180,30,8,15,25,50\n18.5,290,42,14.5,22,23,36\n';
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = '批量预测模板.csv';
    link.click();
}

// ========== 智能工艺建议 ==========

async function loadSuggestions() {
    try {
        const response = await fetch('/api/suggestions');
        const result = await response.json();
        const container = document.getElementById('suggestionList');

        if (!result.success || !result.suggestions || result.suggestions.length === 0) {
            container.innerHTML = '<div class="suggestion-loading">暂无建议</div>';
            return;
        }

        container.innerHTML = result.suggestions.map(s => `
            <div class="suggestion-card ${s.type}">
                <div class="suggestion-title">
                    <span>${s.icon}</span>
                    <span>${s.title}</span>
                </div>
                <div class="suggestion-content">${s.content}</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('加载建议失败:', error);
        document.getElementById('suggestionList').innerHTML =
            '<div class="suggestion-loading">建议加载失败</div>';
    }
}

// ========== 历史记录导出 ==========

function exportHistory() {
    window.open('/api/export', '_blank');
    showToast('正在导出CSV文件...', 'info');
}

// ========== 清空历史记录 ==========

async function clearHistory() {
    if (!confirm('确定要清空所有预测记录吗？此操作不可恢复！')) return;

    try {
        const response = await fetch('/api/history/clear', { method: 'DELETE' });
        const result = await response.json();
        if (result.success) {
            showToast(result.message, 'success');
            loadHistory();
            loadStatistics();
            loadDistributionChart();
        } else {
            showToast(result.error || '清空失败', 'error');
        }
    } catch (error) {
        showToast('网络错误', 'error');
    }
}

// ========== 参数实时校验 ==========

const PARAM_RANGES = {
    arc_voltage: { min: 0, max: 50, name: '电弧电压' },
    welding_current: { min: 0, max: 500, name: '焊接电流' },
    welding_speed: { min: 0, max: 100, name: '焊接速度' },
    wire_feed_speed: { min: 0, max: 30, name: '送丝速度' },
    gas_flow_rate: { min: 0, max: 50, name: '气体流量' },
    torch_angle: { min: 0, max: 90, name: '焊枪角度' },
    base_metal_temperature: { min: 0, max: 200, name: '母材温度' }
};

function validateInput(input) {
    const field = input.id;
    const range = PARAM_RANGES[field];
    if (!range) return true;

    const value = input.value.trim();
    if (value === '') {
        input.classList.remove('invalid');
        return true;
    }

    const num = parseFloat(value);
    if (isNaN(num) || num < range.min || num > range.max) {
        input.classList.add('invalid');
        return false;
    }
    input.classList.remove('invalid');
    return true;
}

function initInputValidation() {
    const fields = ['arc_voltage', 'welding_current', 'welding_speed', 'wire_feed_speed',
                    'gas_flow_rate', 'torch_angle', 'base_metal_temperature'];
    fields.forEach(field => {
        const input = document.getElementById(field);
        if (input) {
            input.addEventListener('input', () => validateInput(input));
            input.addEventListener('blur', () => validateInput(input));
        }
    });
}

// ========== Tab 切换 ==========

function switchTab(tabName) {
    currentTab = tabName;
    // 切换标签样式
    document.querySelectorAll('.tab-item').forEach(item => {
        item.classList.toggle('active', item.dataset.tab === tabName);
    });
    // 切换内容
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === 'tab-' + tabName);
    });
    // 切换到数据分析时重绘图表（解决隐藏div初始化问题）
    if (tabName === 'analysis') {
        setTimeout(() => {
            if (importanceChart) importanceChart.resize();
            if (distributionChart) distributionChart.resize();
        }, 100);
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

    // 绑定应用优化参数按钮
    document.getElementById('btnApplyOptimize').addEventListener('click', applyOptimizeParams);

    // 绑定刷新历史按钮
    document.getElementById('btnRefreshHistory').addEventListener('click', loadHistory);

    // 绑定导出CSV按钮
    document.getElementById('btnExport').addEventListener('click', exportHistory);

    // 绑定清空记录按钮
    document.getElementById('btnClearHistory').addEventListener('click', clearHistory);

    // 初始化参数实时校验
    initInputValidation();

    // 绑定快捷填充按钮
    document.querySelectorAll('.btn-quick').forEach(btn => {
        btn.addEventListener('click', function() {
            fillQuickData(this.dataset.type);
        });
    });

    // 绑定Tab切换
    document.querySelectorAll('.tab-item').forEach(item => {
        item.addEventListener('click', function() {
            switchTab(this.dataset.tab);
        });
    });

    // 绑定预测模式切换
    document.querySelectorAll('.predict-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            switchPredictMode(this.dataset.mode);
        });
    });

    // 初始化批量上传
    initBatchUpload();

    // 初始化加载
    checkHealth();
    loadStatistics();
    loadHistory();
    loadImportanceChart();
    loadDistributionChart();
    loadSuggestions();

    // 定时刷新统计
    setInterval(checkHealth, 30000);
});
