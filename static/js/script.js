/**
 * Guard.AI - Moderation Dashboard Client Controller
 * Production-hardened with error recovery and timeout handling.
 */

const config = {
    apiBase: '/api',
    minTextLength: 5,
    maxTextLength: 10000,
    longInputThreshold: 8000,
    requestTimeoutMs: 120000,  // Increased to 120s for first-run model loading
    healthCheckIntervalMs: 15000  // Check server health every 15s during init
};

let appState = {
    isAnalyzing: false,
    currentResult: null,
    analysisStartTime: null,
    charts: {},
    requestAttempts: 0,
    maxRequestAttempts: 1  // Single attempt per user request (retries managed via UI prompt)
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
const errorCloseBtn = document.querySelector('.error-close');

// Event Listeners
reviewInput.addEventListener('input', handleTextInput);
analyzeBtn.addEventListener('click', handleAnalyze);
clearBtn.addEventListener('click', handleClear);
newAnalysisBtn.addEventListener('click', handleNewAnalysis);
downloadReportBtn.addEventListener('click', handleDownloadReport);

exampleButtons.forEach((btn) => {
    btn.addEventListener('click', (event) => {
        const exampleNum = event.currentTarget.dataset.example;
        reviewInput.value = sampleReviews[exampleNum];
        updateCharCount();
        reviewInput.focus();
    });
});

if (errorCloseBtn) {
    errorCloseBtn.addEventListener('click', hideError);
}

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
    if (appState.isAnalyzing) return;

    const text = reviewInput.value.trim();

    if (text.length < config.minTextLength) {
        showError(`Text must be at least ${config.minTextLength} characters.`);
        return;
    }

    if (text.length > config.maxTextLength) {
        showError('Content exceeds the 10,000 character safety limit. Please shorten the text.');
        return;
    }

    appState.isAnalyzing = true;
    appState.analysisStartTime = performance.now();
    appState.requestAttempts = 0;
    analyzeBtn.disabled = true;
    hideError();
    showLoading();
    resultsSection.classList.add('hidden');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), config.requestTimeoutMs);

    try {
        const response = await fetch(`${config.apiBase}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text }),
            signal: controller.signal
        });

        // Always try to read response as text first to avoid parsing errors
        let responseText = '';
        try {
            responseText = await response.text();
        } catch (readErr) {
            console.error('Failed to read response body:', readErr);
            throw new Error('Failed to read server response');
        }

        // Try to parse JSON, but handle empty responses gracefully
        let result = null;
        if (responseText && responseText.trim()) {
            try {
                result = JSON.parse(responseText);
            } catch (parseError) {
                console.error('JSON parse error:', {
                    statusCode: response.status,
                    responseLength: responseText.length,
                    firstChars: responseText.slice(0, 200)
                });
                throw new Error(`Server returned invalid response (status ${response.status}). Please try again.`);
            }
        }

        // Check HTTP status before validating result content
        if (!response.ok) {
            // Try to extract error message from parsed result
            const errorMsg = (result && result.error) 
                ? result.error
                : `Request failed with status ${response.status}`;
            throw new Error(errorMsg);
        }

        // Ensure result exists and has valid structure
        if (!result) {
            throw new Error('Server returned an empty response. Please retry.');
        }

        if (result.success === false) {
            throw new Error(result.error || 'Analysis failed. Please try again.');
        }

        // Validate result structure before displaying
        if (!validateResultStructure(result)) {
            throw new Error('Server returned incomplete data. Please retry.');
        }

        const processingTime = performance.now() - appState.analysisStartTime;
        appState.currentResult = result;

        hideLoading();
        displayResults(result, processingTime);
    } catch (error) {
        console.error('Analysis error:', error);
        let userMessage = error.message || 'Analysis failed. Please try again.';

        if (error.name === 'AbortError') {
            userMessage = 'Request timed out. Models may still be loading on first run — please wait 30-60 seconds and try again.';
        } else if (error.message && error.message.includes('Failed to fetch')) {
            userMessage = 'Unable to reach the server. Please verify your connection and try again.';
        }

        showError(userMessage);
        hideLoading();
    } finally {
        clearTimeout(timeoutId);
        appState.isAnalyzing = false;
        const length = reviewInput.value.length;
        analyzeBtn.disabled = !(length >= config.minTextLength && length <= config.maxTextLength);
    }
}

function validateResultStructure(result) {
    // Ensure all required fields exist with proper types
    if (!result.toxicity || typeof result.toxicity.confidence !== 'number') return false;
    if (!result.sentiment || typeof result.sentiment.confidence !== 'number') return false;
    if (!result.categories) return false;
    if (typeof result.categories.toxicity !== 'number') return false;
    if (typeof result.categories.hate_speech !== 'number') return false;
    if (typeof result.categories.harassment !== 'number') return false;
    if (typeof result.categories.profanity !== 'number') return false;
    return true;
}

function displayResults(result, processingTime) {
    // Safely extract and validate data with fallbacks
    const isToxic = result.toxicity && result.toxicity.label === 'Toxic';
    const confidence = Math.max(0, Math.min(100, result.toxicity?.confidence || 0));
    const sentiment = result.sentiment || { label: 'Neutral', confidence: 0, scores: {} };

    // 1. Update Toxicity Badge
    resultLabel.textContent = isToxic ? 'Flagged' : 'Safe';
    resultLabel.className = `badge ${isToxic ? 'toxic' : 'safe'}`;

    // 2. Update Sentiment Badge with safe fallback
    const sentimentLabel = (sentiment.label || 'Neutral').toLowerCase();
    sentimentBadge.textContent = (sentiment.label || 'Neutral').charAt(0).toUpperCase() + sentimentLabel.slice(1);
    const sentClass = sentimentLabel === 'positive' ? 'positive' 
                    : sentimentLabel === 'negative' ? 'negative' 
                    : 'neutral';
    sentimentBadge.className = `badge ${sentClass}`;

    // 3. Overall Risk Meter Ring
    confidenceScore.textContent = `${Math.round(confidence)}%`;
    animateCircularProgress(confidence, isToxic);

    // 4. Stats Table with safe defaults
    infoStatus.textContent = isToxic ? 'Action Required' : 'Approved';
    infoStatus.className = `info-value ${isToxic ? 'badge toxic' : 'badge safe'}`;
    infoStatus.style.display = 'inline-flex';
    
    infoRisk.textContent = getRiskLevel(confidence);
    infoTime.textContent = `${Math.round(processingTime)}ms`;
    
    const sentimentConfidence = Math.max(0, Math.min(100, sentiment.confidence || 0));
    sentimentScoreInfo.textContent = `${sentiment.label || 'Neutral'} (${Math.round(sentimentConfidence)}%)`;

    // 5. Individual category bars with safe access
    const categories = result.categories || {
        toxicity: 0,
        hate_speech: 0,
        harassment: 0,
        profanity: 0
    };

    const toxScore = Math.max(0, Math.min(100, categories.toxicity || 0));
    const hateScore = Math.max(0, Math.min(100, categories.hate_speech || 0));
    const harassScore = Math.max(0, Math.min(100, categories.harassment || 0));
    const profScore = Math.max(0, Math.min(100, categories.profanity || 0));

    animateValue(toxicityScore, toxScore);
    animateValue(hateSpeechScore, hateScore);
    animateValue(harassmentScore, harassScore);
    animateValue(profanityScore, profScore);

    animateProgress(toxicityBar, toxScore);
    animateProgress(hateSpeechBar, hateScore);
    animateProgress(harassmentBar, harassScore);
    animateProgress(profanityBar, profScore);

    // Show Results Panel
    resultsSection.classList.remove('hidden');
    
    // Initialize/Render Charts with error recovery
    try {
        updateCharts(result);
    } catch (chartErr) {
        console.error('Chart rendering error:', chartErr);
        // Don't fail the whole result display due to chart issues
    }

    // Scroll to results smoothly
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
    const categories = result.categories || {
        toxicity: 0,
        hate_speech: 0,
        harassment: 0,
        profanity: 0
    };
    const sentiment = result.sentiment || { scores: {} };

    // Destroy existing instances before redraw
    if (appState.charts.categoryChart) {
        try {
            appState.charts.categoryChart.destroy();
        } catch (e) {
            console.warn('Error destroying category chart:', e);
        }
    }

    if (appState.charts.riskChart) {
        try {
            appState.charts.riskChart.destroy();
        } catch (e) {
            console.warn('Error destroying risk chart:', e);
        }
    }

    // Safely get canvas contexts
    const categoryCanvas = document.getElementById('categoryChart');
    const riskCanvas = document.getElementById('riskChart');
    
    if (!categoryCanvas || !riskCanvas) {
        console.warn('Chart canvases not found');
        return;
    }

    const categoryCtx = categoryCanvas.getContext('2d');
    if (!categoryCtx) {
        console.warn('Failed to get category chart context');
        return;
    }

    try {
        appState.charts.categoryChart = new Chart(categoryCtx, {
            type: 'bar',
            data: {
                labels: ['Toxicity', 'Hate Speech', 'Harassment', 'Profanity'],
                datasets: [{
                    data: [
                        Math.max(0, Math.min(100, categories.toxicity || 0)),
                        Math.max(0, Math.min(100, categories.hate_speech || 0)),
                        Math.max(0, Math.min(100, categories.harassment || 0)),
                        Math.max(0, Math.min(100, categories.profanity || 0))
                    ],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.45)',
                        'rgba(245, 158, 11, 0.45)',
                        'rgba(168, 85, 247, 0.45)',
                        'rgba(14, 165, 233, 0.45)'
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
    } catch (err) {
        console.error('Error creating category chart:', err);
    }

    const riskCtx = riskCanvas.getContext('2d');
    if (!riskCtx) {
        console.warn('Failed to get risk chart context');
        return;
    }

    const sScores = sentiment.scores || {};
    const positive = Number(sScores.positive) || 0;
    const neutral = Number(sScores.neutral) || 0;
    const negative = Number(sScores.negative) || 0;

    try {
        appState.charts.riskChart = new Chart(riskCtx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [
                        Math.max(0, Math.min(100, positive)),
                        Math.max(0, Math.min(100, neutral)),
                        Math.max(0, Math.min(100, negative))
                    ],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(245, 158, 11, 0.7)',
                        'rgba(239, 68, 68, 0.7)'
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
    } catch (err) {
        console.error('Error creating risk chart:', err);
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorState.classList.remove('hidden');
}

function hideError() {
    errorState.classList.add('hidden');
}

let statusInterval;

function showLoading() {
    loadingState.classList.remove('hidden');
    const loadingTextElement = loadingState.querySelector('.loading-text');
    if (loadingTextElement) {
        const messages = [
            "Initializing moderation models...",
            "Preparing transformer pipelines...",
            "Running analysis..."
        ];
        let step = 0;
        loadingTextElement.textContent = messages[0];

        clearInterval(statusInterval);
        statusInterval = setInterval(() => {
            step = (step + 1) % messages.length;
            loadingTextElement.textContent = messages[step];
        }, 4000);
    }
}

function hideLoading() {
    loadingState.classList.add('hidden');
    clearInterval(statusInterval);
    const loadingTextElement = loadingState.querySelector('.loading-text');
    if (loadingTextElement) {
        loadingTextElement.textContent = "Performing neural network analysis...";
    }
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
  - Positive:         ${result.sentiment.scores?.positive ?? 0}%
  - Neutral:          ${result.sentiment.scores?.neutral ?? 0}%
  - Negative:         ${result.sentiment.scores?.negative ?? 0}%

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

async function parseJsonResponse(response) {
    const text = await response.text();
    if (!text || !text.trim()) {
        return null;
    }
    try {
        return JSON.parse(text);
    } catch (error) {
        console.error('Invalid JSON response:', {
            statusCode: response.status,
            length: text.length,
            preview: text.slice(0, 200)
        });
        return null;
    }
}

async function init() {
    updateCharCount();

    try {
        const response = await fetch(`${config.apiBase}/health`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });
        const data = await parseJsonResponse(response);

        if (!response.ok || !data) {
            console.warn('Health check failed:', {
                status: response.status,
                hasData: !!data
            });
            showError('Unable to reach the moderation API. Attempting to reconnect...');
            
            // Retry health check after delay
            setTimeout(() => {
                fetch(`${config.apiBase}/health`)
                    .then(r => parseJsonResponse(r))
                    .then(d => {
                        if (d && d.model_loaded) {
                            hideError();
                        } else if (d && !d.model_loaded) {
                            showError('Models are loading in the background. The first analysis may take 30-60 seconds.');
                        }
                    })
                    .catch(() => {
                        console.error('Reconnection failed');
                    });
            }, 3000);
            return;
        }

        if (!data.model_loaded) {
            showError('Models are loading in the background. First analysis may take 30-60 seconds.');
        }
    } catch (err) {
        console.error('Health check exception:', err);
        showError('Unable to connect to moderation server. Please verify the application is running.');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
