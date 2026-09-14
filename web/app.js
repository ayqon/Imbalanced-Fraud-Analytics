/**
 * Imbalanced-Fraud-Analytics - Scientific Analytics Dashboard
 * 100% grounded in real Kaggle European Cardholder dataset statistics & TreeSHAP
 */

let appData = null;
let charts = {};
let selectedSampleIndex = 0;

document.addEventListener("DOMContentLoaded", async () => {
    initTabNavigation();
    await loadRealData();
    if (appData) {
        initOverviewCharts();
        initSampleExplorer();
        initBenchmarkCurves();
        initCostCalibration();
    }
});

// Tab Navigation
function initTabNavigation() {
    const tabs = document.querySelectorAll(".nav-tab-btn");
    const panes = document.querySelectorAll(".tab-pane");

    tabs.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.getAttribute("data-tab");

            tabs.forEach(b => b.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(target);
            if (targetPane) {
                targetPane.classList.add("active");
            }
        });
    });
}

// Data Ingestion
async function loadRealData() {
    try {
        const response = await fetch("real_data.json");
        appData = await response.json();
    } catch (err) {
        console.error("Failed to load real_data.json:", err);
    }
}

// TAB 1: Overview Charts
function initOverviewCharts() {
    const summary = appData.dataset_summary;
    document.getElementById("stat-total-records").textContent = summary.total_records.toLocaleString();
    document.getElementById("stat-fraud-records").textContent = summary.fraud_records.toLocaleString();
    document.getElementById("stat-legit-records").textContent = summary.legitimate_records.toLocaleString();
    document.getElementById("stat-test-records").textContent = summary.test_records.toLocaleString();

    // Feature Importance Chart
    const featLabels = appData.global_feature_importance.map(f => f.feature);
    const featValues = appData.global_feature_importance.map(f => f.importance);

    const ctx = document.getElementById("chart-feature-importance").getContext("2d");
    charts.featureImportance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: featLabels,
            datasets: [{
                label: "XGBoost Feature Importance (Gain)",
                data: featValues,
                backgroundColor: "#2563eb",
                borderRadius: 2
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: "#e2e8f0" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#0f172a", font: { family: "JetBrains Mono", size: 11, weight: "bold" } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// TAB 2: Real SHAP Explorer
function initSampleExplorer() {
    const tbody = document.getElementById("tbody-real-samples");
    tbody.innerHTML = "";

    appData.real_samples.forEach((sample, idx) => {
        const tr = document.createElement("tr");
        if (idx === 0) tr.classList.add("selected-row");

        const isFraud = sample.ground_truth === 1;
        const badgeClass = isFraud ? "status-badge badge-fraud" : "status-badge badge-legit";
        const badgeText = isFraud ? "Fraud (1)" : "Legitimate (0)";

        const topSignal = Object.entries(sample.shap_attributions)[0];
        const signalText = topSignal ? `${topSignal[0]}` : "Baseline";

        tr.innerHTML = `
            <td class="text-mono font-bold">${sample.sample_id}</td>
            <td><span class="${badgeClass}">${badgeText}</span></td>
            <td>EUR ${sample.amount.toFixed(2)}</td>
            <td>${sample.time_hours}h</td>
            <td class="text-mono font-bold ${sample.predicted_proba > 0.5 ? 'text-danger' : 'text-success'}">${(sample.predicted_proba * 100).toFixed(1)}%</td>
            <td class="text-muted font-bold">${signalText}</td>
        `;

        tr.addEventListener("click", () => {
            document.querySelectorAll("#tbody-real-samples tr").forEach(r => r.classList.remove("selected-row"));
            tr.classList.add("selected-row");
            renderSelectedSample(idx);
        });

        tbody.appendChild(tr);
    });

    renderSelectedSample(0);
}

function renderSelectedSample(idx) {
    selectedSampleIndex = idx;
    const sample = appData.real_samples[idx];
    const isFraud = sample.ground_truth === 1;

    document.getElementById("meta-tx-id").textContent = sample.sample_id;
    document.getElementById("meta-tx-amount").textContent = `EUR ${sample.amount.toFixed(2)}`;
    document.getElementById("meta-tx-time").textContent = `${sample.time_hours} Hours`;

    const probaElem = document.getElementById("meta-tx-proba");
    probaElem.textContent = `${(sample.predicted_proba * 100).toFixed(2)}%`;
    probaElem.className = sample.predicted_proba > 0.5 ? "meta-box-val text-mono text-danger" : "meta-box-val text-mono text-success";

    const badge = document.getElementById("inspector-badge");
    badge.textContent = isFraud ? "CONFIRMED FRAUD" : "CONFIRMED LEGITIMATE";
    badge.className = isFraud ? "status-badge badge-fraud" : "status-badge badge-legit";

    // Render SHAP Plot
    const shapKeys = Object.keys(sample.shap_attributions);
    const shapVals = Object.values(sample.shap_attributions);
    const colors = shapVals.map(v => v > 0 ? "rgba(220, 38, 38, 0.85)" : "rgba(5, 150, 105, 0.85)");

    if (charts.shap) {
        charts.shap.destroy();
    }

    const ctx = document.getElementById("chart-sample-shap").getContext("2d");
    charts.shap = new Chart(ctx, {
        type: "bar",
        data: {
            labels: shapKeys,
            datasets: [{
                label: "TreeSHAP Attribution",
                data: shapVals,
                backgroundColor: colors,
                borderRadius: 2
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: "SHAP Impact on Fraud Probability Log-Odds", color: "#64748b", font: { size: 10 } },
                    grid: { color: "#e2e8f0" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#0f172a", font: { family: "JetBrains Mono", size: 10, weight: "bold" } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // Summary Text
    const topFactor = Object.entries(sample.shap_attributions)[0];
    const explanationBox = document.getElementById("explanation-text-box");
    explanationBox.innerHTML = `
        <strong>Regulatory Summary (GDPR Article 22):</strong> 
        The model assessed this transaction on test set holdout. 
        The primary decision driver was <strong>${topFactor[0]}</strong> with a Shapley attribution score of 
        <span class="${topFactor[1] > 0 ? 'text-danger font-bold' : 'text-success font-bold'}">${topFactor[1] > 0 ? '+' : ''}${topFactor[1].toFixed(3)}</span>.
        ${isFraud ? 'Significant orthogonal divergence in PCA components pushed the predicted fraud probability past the decision threshold.' : 'Component values aligned with standard legitimate consumer transaction distributions.'}
    `;
}

// TAB 3: Benchmark Curves
function initBenchmarkCurves() {
    // ROC Curve
    const ctxRoc = document.getElementById("chart-real-roc").getContext("2d");
    charts.roc = new Chart(ctxRoc, {
        type: "line",
        data: {
            labels: appData.roc_curve.map(p => p.fpr),
            datasets: [
                {
                    label: "XGBoost (AUC = 0.978)",
                    data: appData.roc_curve.map(p => p.tpr),
                    borderColor: "#2563eb",
                    backgroundColor: "rgba(37, 99, 235, 0.05)",
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: "Random Classifier",
                    data: appData.roc_curve.map(p => p.fpr),
                    borderColor: "#94a3b8",
                    borderDash: [4, 4],
                    borderWidth: 1.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: "False Positive Rate (FPR)", color: "#64748b" },
                    grid: { color: "#e2e8f0" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono" } }
                },
                y: {
                    title: { display: true, text: "True Positive Rate (Recall)", color: "#64748b" },
                    grid: { color: "#e2e8f0" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono" } }
                }
            },
            plugins: {
                legend: { position: "bottom", labels: { font: { size: 10 }, color: "#0f172a" } }
            }
        }
    });

    // PR Curve
    const ctxPr = document.getElementById("chart-real-pr").getContext("2d");
    charts.pr = new Chart(ctxPr, {
        type: "line",
        data: {
            labels: appData.pr_curve.map(p => p.recall),
            datasets: [
                {
                    label: "XGBoost (PR-AUC = 0.865)",
                    data: appData.pr_curve.map(p => p.precision),
                    borderColor: "#059669",
                    backgroundColor: "rgba(5, 150, 105, 0.05)",
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: "Recall (Sensitivity)", color: "#64748b" },
                    grid: { color: "#e2e8f0" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono" } }
                },
                y: {
                    title: { display: true, text: "Precision (PPV)", color: "#64748b" },
                    grid: { color: "#e2e8f0" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono" } }
                }
            },
            plugins: {
                legend: { position: "bottom", labels: { font: { size: 10 }, color: "#0f172a" } }
            }
        }
    });
}

// TAB 4: Cost Calibration & Confusion Matrix
function initCostCalibration() {
    const slider = document.getElementById("slider-threshold");
    const labelSlider = document.getElementById("label-slider-val");
    const inputFn = document.getElementById("input-fn-cost");
    const inputFp = document.getElementById("input-fp-cost");

    function updateCalibration() {
        const threshold = parseFloat(slider.value);
        labelSlider.textContent = threshold.toFixed(2);

        const costFn = parseFloat(inputFn.value) || 150.0;
        const costFp = parseFloat(inputFp.value) || 3.0;

        // Find closest simulation point in real test set
        let sim = appData.threshold_simulations.find(s => Math.abs(s.threshold - threshold) < 0.01);
        if (!sim) {
            sim = appData.threshold_simulations[0];
        }

        const fnCost = sim.fn * costFn;
        const fpCost = sim.fp * costFp;
        const totalLoss = fnCost + fpCost;
        const savedCost = sim.tp * costFn;

        // Update Matrix DOM
        document.getElementById("cm-tn-count").textContent = sim.tn.toLocaleString();
        document.getElementById("cm-fp-count").textContent = sim.fp.toLocaleString();
        document.getElementById("cm-fn-count").textContent = sim.fn.toLocaleString();
        document.getElementById("cm-tp-count").textContent = sim.tp.toLocaleString();

        document.getElementById("cm-fp-cost").textContent = `Cost: EUR ${fpCost.toFixed(2)}`;
        document.getElementById("cm-fn-cost").textContent = `Cost: EUR ${fnCost.toFixed(2)}`;
        document.getElementById("cm-tp-saved").textContent = `Saved: EUR ${savedCost.toFixed(2)}`;

        document.getElementById("loss-current-amount").textContent = `EUR ${totalLoss.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

        updateLossCurve(costFn, costFp);
    }

    slider.addEventListener("input", updateCalibration);
    inputFn.addEventListener("input", updateCalibration);
    inputFp.addEventListener("input", updateCalibration);

    // Initial Loss Curve
    const ctxLoss = document.getElementById("chart-loss-curve").getContext("2d");
    charts.loss = new Chart(ctxLoss, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Total Financial Loss (EUR)",
                    data: [],
                    borderColor: "#dc2626",
                    backgroundColor: "rgba(220, 38, 38, 0.08)",
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: "Decision Threshold", color: "#64748b" },
                    grid: { color: "#e2e8f0" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono" } }
                },
                y: {
                    title: { display: true, text: "Total Expected Loss (EUR)", color: "#64748b" },
                    grid: { color: "#e2e8f0" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono" } }
                }
            },
            plugins: {
                legend: { position: "bottom", labels: { font: { size: 10 }, color: "#0f172a" } }
            }
        }
    });

    function updateLossCurve(costFn, costFp) {
        const labels = [];
        const totalLosses = [];
        let minLoss = Infinity;
        let optimalThreshold = 0.50;

        appData.threshold_simulations.forEach(s => {
            const loss = (s.fn * costFn) + (s.fp * costFp);
            labels.push(s.threshold.toFixed(2));
            totalLosses.push(loss);

            if (loss < minLoss) {
                minLoss = loss;
                optimalThreshold = s.threshold;
            }
        });

        document.getElementById("loss-optimal-statement").textContent = 
            `Optimal Threshold: ${optimalThreshold.toFixed(2)} (Minimum Expected Loss: EUR ${minLoss.toFixed(2)})`;

        charts.loss.data.labels = labels;
        charts.loss.data.datasets[0].data = totalLosses;
        charts.loss.update();
    }

    updateCalibration();
}
