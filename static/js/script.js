/**
 * Guard.AI - Moderation Dashboard Client Controller
 */

console.log("NEW SCRIPT VERSION LOADED");

const config = {
    apiBase: '/api',
    minTextLength: 5,
    maxTextLength: 10000,
    longInputThreshold: 8000
};

let appState = {
    isAnalyzing: false,
    currentResult: null,
    analysisStartTime: null,
    charts: {}
};

const sampleReviews = {
    1: "The product arrived on time and matched the description. Support handled my question quickly.",
    2: "The item stopped working after one week. I expected better quality for the price.",
    3: "This is useless trash and the seller should be ashamed of this garbage product."
};

// UI Element Selections
const reviewInput = document.getElementById('review-input');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const charCount = document.querySelector('.char-count');
const loadingState = document.getElementById('loading-state');
const errorState = document.getElementById('error-state');
const errorMessage = document.getElementById('error-message');
const resultsSection = document.getElementById('results-section');

// Summary Cards
const resultLabel = document.getElementById('result-label');
const sentimentBadge = document.getElementById('sentiment-badge');
const confidenceScore = document.getElementById('confidence-score');
const progressCircle = document.getElementById('progress-circle');

const infoStatus = document.getElementById('info-status');
const infoRisk = document.getElementById('info-risk');
const sentimentScoreInfo = document.getElementById('sentiment-score-info');
const infoTime = document.getElementById('info-time');

// Category Scores & Progress Bars
const toxicityScore = document.getElementById('toxicity-score');
const hateSpeechScore = document.getElementById('hate-speech-score');
const harassmentScore = document.getElementById('harassment-score');
const profanityScore = document.getElementById('profanity-score');

const toxicityBar = document.getElementById('toxicity-bar');
const hateSpeechBar = document.getElementById('hate-speech-bar');
const harassmentBar = document.getElementById('harassment-bar');
const profanityBar = document.getElementById('profanity-bar');

const newAnalysisBtn = document.getElementById('new-analysis-btn');
const downloadReportBtn = document.getElementById('download-report-btn');
const exampleButtons = document.querySelectorAll('.example-btn');

// Event Listeners
reviewInput.addEventListener('input', handleTextInput);
analyzeBtn.addEventListener('click', handleAnalyze);
clearBtn.addEventListener('click', handleClear);
newAnalysisBtn.addEventListener('click', handleNewAnalysis);
downloadReportBtn.addEventListener('click', handleDownloadReport);

exampleButtons.forEach((btn) => {
    btn.addEventListener('click', (event) => {
        const exampleNum = event.target.dataset.example;
        reviewInput.value = sampleReviews[exampleNum];
        updateCharCount();
        reviewInput.focus();
    });
});

function handleTextInput() {
    updateCharCount();
}

function updateCharCount() {
    const length = reviewInput.value.length;
    charCount.classList.remove('warning', 'limit');

    if (length > config.maxTextLength) {
        charCount.textContent = 'Large moderation payload detected';
        charCount.classList.add('limit');
    } else if (length >= config.longInputThreshold) {
        charCount.textContent = `${length.toLocaleString()} characters - long input`;
        charCount.classList.add('warning');
    } else {
        charCount.textContent = `${length.toLocaleString()} characters`;
    }

    const isValid = length >= config.minTextLength && length <= config.maxTextLength;
    analyzeBtn.disabled = !isValid;
}

async function handleAnalyze() {
    const text = reviewInput.value.trim();

    if (text.length < config.minTextLength) {
        showError(`Text must be at least ${config.minTextLength} characters.`);
        return;
    }

    if (text.length > config.maxTextLength) {
        showError('Large moderation payload detected. Please shorten the content before analysis.');
        return;
    }

    appState.isAnalyzing = true;
    appState.analysisStartTime = performance.now();
    analyzeBtn.disabled = true;
    hideError();
    showLoading();
    resultsSection.classList.add('hidden');

    try {
        const response = await fetch(`${config.apiBase}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || `Request failed with status ${response.status}`);
        }

        if (!result.success) {
            throw new Error(result.error || 'Moderation analysis failed.');
        }

        const processingTime = performance.now() - appState.analysisStartTime;
        appState.currentResult = result;

        hideLoading();
        displayResults(result, processingTime);
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'Analysis failed. Please try again.');
        hideLoading();
    } finally {
        appState.isAnalyzing = false;
        analyzeBtn.disabled = false;
    }
}

function displayResults(result, processingTime) {
    const isToxic = result.toxicity.label === 'Toxic';
    const confidence = result.toxicity.confidence;
    const sentiment = result.sentiment;

    // 1. Update Toxicity Badge
    resultLabel.textContent = isToxic ? 'Flagged' : 'Safe';
    resultLabel.className = `badge ${isToxic ? 'toxic' : 'safe'}`;

    // 2. Update Sentiment Badge
    sentimentBadge.textContent = sentiment.label;
    const sentClass = sentiment.label.toLowerCase();
    sentimentBadge.className = `badge ${sentClass === 'positive' ? 'positive' : sentClass === 'negative' ? 'negative' : 'neutral'}`;

    // 3. Overall Risk Meter Ring
    confidenceScore.textContent = `${Math.round(confidence)}%`;
    animateCircularProgress(confidence, isToxic);

    // 4. Stats Table
    infoStatus.textContent = isToxic ? 'Action Required' : 'Approved';
    infoStatus.className = `info-value ${isToxic ? 'badge toxic' : 'badge safe'}`;
    infoStatus.style.display = 'inline-flex';
    
    infoRisk.textContent = getRiskLevel(confidence);
    infoTime.textContent = `${Math.round(processingTime)}ms`;
    sentimentScoreInfo.textContent = `${sentiment.label} (${Math.round(sentiment.confidence)}%)`;

    // 5. Individual category bars
    animateValue(toxicityScore, result.categories.toxicity);
    animateValue(hateSpeechScore, result.categories.hate_speech);
    animateValue(harassmentScore, result.categories.harassment);
    animateValue(profanityScore, result.categories.profanity);

    animateProgress(toxicityBar, result.categories.toxicity);
    animateProgress(hateSpeechBar, result.categories.hate_speech);
    animateProgress(harassmentBar, result.categories.harassment);
    animateProgress(profanityBar, result.categories.profanity);

    // Show Results Panel
    resultsSection.classList.remove('hidden');
    
    // Initialize/Render Charts
    updateCharts(result);

    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 150);
}

function getRiskLevel(confidence) {
    if (confidence >= 75) return 'Critical';
    if (confidence >= 50) return 'High';
    if (confidence >= 25) return 'Medium';
    return 'Low';
}

function animateValue(element, targetValue) {
    const startValue = 0;
    const duration = 800;
    const startTime = performance.now();

    function updateValue(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const value = Math.round(startValue + (targetValue - startValue) * easeOutCubic(progress));
        element.textContent = `${value}%`;

        if (progress < 1) {
            requestAnimationFrame(updateValue);
        }
    }

    requestAnimationFrame(updateValue);
}

function animateProgress(element, targetValue) {
    const duration = 1000;
    const startTime = performance.now();

    // Set bar color based on category value
    if (targetValue >= 75) {
        element.style.backgroundColor = 'var(--color-danger)';
    } else if (targetValue >= 35) {
        element.style.backgroundColor = 'var(--color-warning)';
    } else {
        element.style.backgroundColor = 'var(--color-safe)';
    }

    function updateProgress(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const value = targetValue * easeOutCubic(progress);
        element.style.width = `${value}%`;

        if (progress < 1) {
            requestAnimationFrame(updateProgress);
        }
    }

    requestAnimationFrame(updateProgress);
}

function animateCircularProgress(targetValue, isToxic) {
    const circumference = 534; // 2 * pi * r (r=85) -> 534.07
    const offset = circumference - (targetValue / 100) * circumference;
    const duration = 1200;
    const startTime = performance.now();
    const startOffset = circumference;

    // Set color based on risk levels
    if (isToxic) {
        progressCircle.style.stroke = 'var(--color-danger)';
    } else if (targetValue >= 25) {
        progressCircle.style.stroke = 'var(--color-warning)';
    } else {
        progressCircle.style.stroke = 'var(--color-safe)';
    }

    function updateCircle(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const currentOffset = startOffset + (offset - startOffset) * easeOutCubic(progress);
        progressCircle.style.strokeDashoffset = currentOffset;

        if (progress < 1) {
            requestAnimationFrame(updateCircle);
        }
    }

    requestAnimationFrame(updateCircle);
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

function updateCharts(result) {
    const categories = result.categories;
    const sentiment = result.sentiment;

    // Destroy existing instances before redraw
    if (appState.charts.categoryChart) {
        appState.charts.categoryChart.destroy();
    }

    if (appState.charts.riskChart) {
        appState.charts.riskChart.destroy();
    }

    const categoryCtx = document.getElementById('categoryChart').getContext('2d');
    appState.charts.categoryChart = new Chart(categoryCtx, {
        type: 'bar',
        data: {
            labels: ['Toxicity', 'Hate Speech', 'Harassment', 'Profanity'],
            datasets: [{
                data: [
                    categories.toxicity,
                    categories.hate_speech,
                    categories.harassment,
                    categories.profanity
                ],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.45)',  // red
                    'rgba(245, 158, 11, 0.45)',  // amber
                    'rgba(168, 85, 247, 0.45)',  // purple
                    'rgba(14, 165, 233, 0.45)'   // cyan
                ],
                borderColor: [
                    '#ef4444',
                    '#f59e0b',
                    '#a855f7',
                    '#0ea5e9'
                ],
                borderWidth: 1.5,
                borderRadius: 4,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#64748b',
                        font: { family: "'Inter', sans-serif", size: 10 }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8',
                        font: { family: "'Inter', sans-serif", size: 11, weight: '500' }
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    const riskCtx = document.getElementById('riskChart').getContext('2d');
    const sScores = sentiment.scores || {};
    
    appState.charts.riskChart = new Chart(riskCtx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [
                    sScores.positive || 0,
                    sScores.neutral || 0,
                    sScores.negative || 0
                ],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.7)', // green
                    'rgba(245, 158, 11, 0.7)', // amber
                    'rgba(239, 68, 68, 0.7)'  // red
                ],
                borderColor: '#0b0f19',
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#94a3b8',
                        font: { family: "'Inter', sans-serif", size: 11 },
                        padding: 12,
                        boxWidth: 8,
                        boxHeight: 8,
                        usePointStyle: true
                    }
                }
            }
        }
    });
}

function showError(message) {
    errorMessage.textContent = message;
    errorState.classList.remove('hidden');
}

function hideError() {
    errorState.classList.add('hidden');
}

function showLoading() {
    loadingState.classList.remove('hidden');
}

function hideLoading() {
    loadingState.classList.add('hidden');
}

function resetWorkspace({ focusInput = false } = {}) {
    reviewInput.value = '';
    if (focusInput) {
        reviewInput.focus();
    }
    updateCharCount();
    resultsSection.classList.add('hidden');
    hideError();
    appState.currentResult = null;
}

function handleClear() {
    resetWorkspace();
}

function handleNewAnalysis() {
    resetWorkspace({ focusInput: true });
}

function handleDownloadReport() {
    if (!appState.currentResult) return;

    const result = appState.currentResult;
    const timestamp = new Date(result.timestamp).toLocaleString();

    const reportContent = `GUARD.AI - CONTENT MODERATION REPORT
==================================================
Date generated: ${timestamp}
Review Content:
--------------------------------------------------
"${reviewInput.value}"

==================================================
COMPLIANCE SIGNALS
==================================================
Toxicity Assessment:  ${result.toxicity.label.toUpperCase()}
Risk Score:           ${result.toxicity.confidence}%
Moderation Status:    ${result.toxicity.label === 'Toxic' ? 'ACTION REQUIRED' : 'COMPLIANT'}
Risk Level:           ${getRiskLevel(result.toxicity.confidence)}

Sentiment Quality:    ${result.sentiment.label} (${result.sentiment.confidence}%)
Sentiment Scores:
  - Positive:         ${result.sentiment.scores.positive}%
  - Neutral:          ${result.sentiment.scores.neutral}%
  - Negative:         ${result.sentiment.scores.negative}%

==================================================
DETAILED CATEGORIES PROBABILITY
==================================================
- Toxicity:           ${result.categories.toxicity}%
- Hate Speech:        ${result.categories.hate_speech}%
- Harassment:         ${result.categories.harassment}%
- Profanity:          ${result.categories.profanity}%

==================================================
End of Guard.AI Compliance Log. processed locally.
`;

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `guard_ai_audit_log_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function init() {
    console.log('Guard.AI Engine initialized.');
    updateCharCount();

    fetch(`${config.apiBase}/health`)
        .then((res) => res.json())
        .then((data) => {
            console.log('Guard.AI Backend health OK', data);
            if (!data.model_loaded) {
                showError('Moderation models are warming up in the background. First requests may take a minute.');
            }
        })
        .catch((err) => {
            console.error('Guard.AI Health check failed', err);
            showError('Unable to connect to local Guard.AI server. Please verify application runtime.');
        });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
