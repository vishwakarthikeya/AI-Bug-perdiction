// Developer Mode JavaScript
class DeveloperAnalyzer {
    constructor() {
        this.apiBaseUrl = 'http://127.0.0.1:8000';
        this.history = this.loadHistory();
        this.currentAnalysis = null;
        
        this.init();
    }

    init() {
        // Initialize Chart.js
        this.initChart();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Load example
        this.loadExample();
        
        // Render history
        this.renderHistory();
    }

    setupEventListeners() {
        // Analyze button
        document.querySelector('.analyze-btn')?.addEventListener('click', () => {
            this.analyzeCode();
        });

        // Fix button
        document.querySelector('.fix-btn')?.addEventListener('click', () => {
            this.getAIHelp();
        });

        // Clear button
        document.querySelector('.clear-btn')?.addEventListener('click', () => {
            this.clearCode();
        });

        // Example button
        document.querySelector('.example-btn')?.addEventListener('click', () => {
            this.loadExample();
        });

        // Language selector
        document.getElementById('languageSelect')?.addEventListener('change', (e) => {
            this.updateLanguage(e.target.value);
        });

        // Code input changes
        const codeInput = document.getElementById('codeInput');
        if (codeInput) {
            codeInput.addEventListener('input', Utils.debounce(() => {
                this.updateCodeMetrics();
            }, 500));
        }
    }

    async analyzeCode() {
        const code = document.getElementById('codeInput')?.value;
        if (!code?.trim()) {
            Utils.showNotification('Please enter some code to analyze', 'error');
            return;
        }

        // Show loading
        this.showLoading();

        try {
            // Send to bug prediction endpoint
            const response = await fetch(`${this.apiBaseUrl}/predict-bug`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code })
            });

            const data = await response.json();
            
            // Hide loading
            this.hideLoading();
            
            // Update UI with results
            this.updateAnalysisResults(data);
            
            // Save to history
            this.saveToHistory({
                code: code.substring(0, 100) + (code.length > 100 ? '...' : ''),
                timestamp: new Date(),
                result: data
            });

        } catch (error) {
            console.error('Error:', error);
            this.hideLoading();
            
            // Fallback to local analysis
            this.performLocalAnalysis(code);
            
            Utils.showNotification('Using local analysis (backend unavailable)', 'warning');
        }
    }

    async getAIHelp() {
        const code = document.getElementById('codeInput')?.value;
        if (!code?.trim()) {
            Utils.showNotification('Please enter some code first', 'error');
            return;
        }

        // Show loading
        this.showLoading();

        try {
            // Send to AI fix endpoint
            const response = await fetch(`${this.apiBaseUrl}/developer-fix`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ code })
            });

            const data = await response.json();
            
            // Hide loading
            this.hideLoading();
            
            // Update UI with AI fix
            this.updateAIFix(data);

        } catch (error) {
            console.error('Error:', error);
            this.hideLoading();
            
            // Show fallback fix
            this.showFallbackFix(code);
            
            Utils.showNotification('Showing fallback fix (AI service unavailable)', 'warning');
        }
    }

    updateAnalysisResults(data) {
        // Update bug probability
        const probability = (data.probability * 100).toFixed(1);
        document.getElementById('bugProbability').textContent = `${probability}%`;
        
        // Update progress bar
        const progressFill = document.getElementById('bugProgress');
        if (progressFill) {
            progressFill.style.width = `${probability}%`;
        }
        
        // Update severity badge
        const severity = data.severity || 'low';
        const severityBadge = document.getElementById('severityBadge');
        if (severityBadge) {
            severityBadge.textContent = severity.toUpperCase();
            severityBadge.className = 'severity-badge';
            severityBadge.classList.add(severity);
        }
        
        // Update bug analysis
        const analysisDiv = document.getElementById('bugAnalysis');
        if (analysisDiv && data.explanation) {
            analysisDiv.textContent = data.explanation;
        }
        
        // Update metrics
        this.updateMetrics(data);
        
        // Update chart
        this.updateChart(data);
        
        // Store current analysis
        this.currentAnalysis = data;
    }

    updateMetrics(data) {
        // Lines of code
        const loc = data.metrics?.loc || this.countLinesOfCode();
        document.getElementById('locCount').textContent = loc;
        
        // Complexity score
        const complexity = data.metrics?.complexity || this.calculateComplexity();
        document.getElementById('complexityScore').textContent = complexity;
        
        // Issue count
        const issues = data.metrics?.issues || this.countIssues();
        document.getElementById('issueCount').textContent = issues;
    }

    updateAIFix(data) {
        const fixedCodeDiv = document.getElementById('fixedCode');
        const explanationDiv = document.getElementById('fixExplanation');
        
        if (fixedCodeDiv && data.fixed_code) {
            fixedCodeDiv.textContent = data.fixed_code;
            this.highlightCode(fixedCodeDiv);
        }
        
        if (explanationDiv && data.explanation) {
            explanationDiv.textContent = data.explanation;
        }
    }

    updateCodeMetrics() {
        const code = document.getElementById('codeInput')?.value || '';
        
        // Update LOC
        const loc = this.countLinesOfCode(code);
        document.getElementById('locCount').textContent = loc;
        
        // Update complexity
        const complexity = this.calculateComplexity(code);
        document.getElementById('complexityScore').textContent = complexity;
    }

    countLinesOfCode(code = null) {
        if (!code) {
            code = document.getElementById('codeInput')?.value || '';
        }
        return code.split('\n').filter(line => line.trim().length > 0).length;
    }

    calculateComplexity(code = null) {
        if (!code) {
            code = document.getElementById('codeInput')?.value || '';
        }
        
        let complexity = 0;
        
        // Count control structures
        const patterns = [
            /\bif\s*\(/g,
            /\bfor\s*\(/g,
            /\bwhile\s*\(/g,
            /\bswitch\s*\(/g,
            /\btry\s*{/g,
            /\bcatch\s*\(/g,
            /\b&&|\|\|/g,
            /\?.*:/g
        ];
        
        patterns.forEach(pattern => {
            const matches = code.match(pattern);
            if (matches) complexity += matches.length;
        });
        
        return complexity;
    }

    countIssues() {
        const code = document.getElementById('codeInput')?.value || '';
        let issues = 0;
        
        // Simple pattern matching for common issues
        const issuePatterns = [
            /\/\s*[0-9]/,  // Division by number (potential division by zero)
            /null\b/,       // null references
            /undefined/,    // undefined
            /\.length\s*[<=>]/  // Potential off-by-one
        ];
        
        issuePatterns.forEach(pattern => {
            const matches = code.match(pattern);
            if (matches) issues += matches.length;
        });
        
        return issues;
    }

    initChart() {
        const ctx = document.getElementById('bugChart')?.getContext('2d');
        if (!ctx) return;

        this.chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Critical', 'High', 'Medium', 'Low'],
                datasets: [{
                    data: [10, 20, 30, 40],
                    backgroundColor: [
                        '#ff4444',
                        '#ffaa00',
                        '#ffff00',
                        '#00ff88'
                    ],
                    borderWidth: 2,
                    borderColor: '#0f0c29'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    updateChart(data) {
        if (!this.chart) return;

        const probability = data.probability || 0.5;
        const severity = data.severity || 'medium';
        
        // Update chart data based on severity
        const severityMap = {
            'critical': [80, 10, 5, 5],
            'high': [10, 70, 10, 10],
            'medium': [5, 15, 60, 20],
            'low': [2, 8, 20, 70]
        };
        
        const distribution = severityMap[severity] || [25, 25, 25, 25];
        
        this.chart.data.datasets[0].data = distribution;
        this.chart.update();
    }

    clearCode() {
        if (confirm("Clear the code editor?")) {
            document.getElementById('codeInput').value = '';
            this.resetResults();
        }
    }

    resetResults() {
        document.getElementById('bugProbability').textContent = '0%';
        document.getElementById('bugProgress').style.width = '0%';
        document.getElementById('severityBadge').textContent = 'Not Analyzed';
        document.getElementById('severityBadge').className = 'severity-badge';
        document.getElementById('bugAnalysis').textContent = 'Submit code to see detailed analysis...';
        document.getElementById('fixedCode').textContent = '// AI-generated fix will appear here';
        document.getElementById('fixExplanation').textContent = 'Click "Get AI Fix" for explanation';
        
        // Reset metrics
        this.updateCodeMetrics();
        document.getElementById('issueCount').textContent = '0';
        
        // Reset chart
        if (this.chart) {
            this.chart.data.datasets[0].data = [25, 25, 25, 25];
            this.chart.update();
        }
    }

    loadExample() {
        const examples = {
            python: `def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    average = total / len(numbers)  # Potential division by zero
    return average

def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] is not None:
            value = data[i] * 2
            result.append(value)
    return result

def main():
    # Test with empty list
    numbers = []
    avg = calculate_average(numbers)
    print(f"Average: {avg}")
    
    # Test with None values
    data = [1, 2, None, 4, 5]
    processed = process_data(data)
    print(f"Processed: {processed}")

if __name__ == "__main__":
    main()`,
            
            javascript: `function processUserData(users) {
    let totalAge = 0;
    
    for (let i = 0; i <= users.length; i++) {  // Off-by-one error
        totalAge += users[i].age;  // Potential null access
    }
    
    const averageAge = totalAge / users.length;  // Division by zero
    
    return {
        average: averageAge,
        summary: \`Average age is \${averageAge}\`
    };
}

// Missing null check
const data = null;
const result = processUserData(data);`,
            
            java: `public class DataProcessor {
    public double calculateAverage(List<Integer> numbers) {
        int sum = 0;
        for (int num : numbers) {
            sum += num;
        }
        return sum / numbers.size();  // Integer division
    }
    
    public void process() {
        List<Integer> data = null;
        double avg = calculateAverage(data);  // Null pointer
        System.out.println("Average: " + avg);
    }
}`
        };
        
        const lang = document.getElementById('languageSelect')?.value || 'python';
        document.getElementById('codeInput').value = examples[lang] || examples.python;
        
        // Update metrics for example
        this.updateCodeMetrics();
    }

    updateLanguage(lang) {
        // Update code example based on language
        this.loadExample();
    }

    showLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'flex';
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    performLocalAnalysis(code) {
        // Simple local analysis when backend is unavailable
        const analysis = {
            probability: this.estimateBugProbability(code),
            severity: this.determineSeverity(code),
            explanation: this.generateLocalExplanation(code),
            metrics: {
                loc: this.countLinesOfCode(code),
                complexity: this.calculateComplexity(code),
                issues: this.countIssues()
            }
        };
        
        this.updateAnalysisResults(analysis);
    }

    estimateBugProbability(code) {
        let score = 0;
        
        // Check for common bug patterns
        const patterns = [
            { pattern: /\/\s*0/, weight: 0.9 },  // Division by zero
            { pattern: /null\b/, weight: 0.7 },   // null references
            { pattern: /undefined/, weight: 0.6 }, // undefined
            { pattern: /\.length\s*[<=>]/, weight: 0.5 }, // Array bounds
            { pattern: /\bfor\s*\([^)]*<=/, weight: 0.4 }, // Potential off-by-one
            { pattern: /\btry\s*{/, weight: 0.3 }  // Exception handling
        ];
        
        patterns.forEach(({ pattern, weight }) => {
            const matches = code.match(pattern);
            if (matches) score += matches.length * weight;
        });
        
        // Normalize to 0-1 range
        return Math.min(0.95, score / 10);
    }

    determineSeverity(code) {
        const probability = this.estimateBugProbability(code);
        
        if (probability > 0.8) return 'critical';
        if (probability > 0.6) return 'high';
        if (probability > 0.4) return 'medium';
        return 'low';
    }

    generateLocalExplanation(code) {
        const issues = [];
        
        if (code.includes('/ 0')) {
            issues.push("• Division by zero detected");
        }
        if (code.includes('null')) {
            issues.push("• Potential null pointer references");
        }
        if (code.includes('undefined')) {
            issues.push("• Undefined variable access");
        }
        if (code.match(/\.length\s*[<=>]/)) {
            issues.push("• Possible array index out of bounds");
        }
        
        if (issues.length === 0) {
            return "No obvious bugs detected. Code appears clean.";
        }
        
        return "Potential issues found:\n" + issues.join('\n');
    }

    showFallbackFix(code) {
        // Simple fallback fix demonstration
        const fixes = {
            python: `def calculate_average(numbers):
    if not numbers:  # Added check for empty list
        return 0
    total = 0
    for num in numbers:
        total += num
    average = total / len(numbers)
    return average

def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] is not None:
            value = data[i] * 2
            result.append(value)
    return result

def main():
    numbers = []
    avg = calculate_average(numbers)
    print(f"Average: {avg}")
    
    data = [1, 2, None, 4, 5]
    processed = process_data(data)
    print(f"Processed: {processed}")

if __name__ == "__main__":
    main()`
        };
        
        const lang = document.getElementById('languageSelect')?.value || 'python';
        const fixedCode = fixes[lang] || "// AI fix unavailable. Check your code for common bugs.";
        
        document.getElementById('fixedCode').textContent = fixedCode;
        document.getElementById('fixExplanation').textContent = "Added null checks and input validation to prevent common runtime errors.";
    }

    saveToHistory(item) {
        this.history.unshift(item);
        if (this.history.length > 10) {
            this.history.pop();
        }
        
        this.renderHistory();
        this.saveHistory();
    }

    renderHistory() {
        const container = document.getElementById('historyList');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.history.forEach((item, index) => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.onclick = () => this.loadFromHistory(index);
            
            const time = new Date(item.timestamp).toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            const severity = item.result?.severity || 'low';
            
            div.innerHTML = `
                <h4>Analysis ${index + 1}</h4>
                <p>${item.code}</p>
                <div class="severity ${severity}">${severity.toUpperCase()}</div>
                <small>${time}</small>
            `;
            
            container.appendChild(div);
        });
    }

    loadFromHistory(index) {
        if (index >= 0 && index < this.history.length) {
            const item = this.history[index];
            this.updateAnalysisResults(item.result);
            
            // Highlight the selected item
            document.querySelectorAll('.history-item').forEach((el, i) => {
                el.style.background = i === index ? 'rgba(0, 219, 222, 0.2)' : '';
            });
        }
    }

    loadHistory() {
        try {
            const saved = localStorage.getItem('developerHistory');
            return saved ? JSON.parse(saved) : [];
        } catch (e) {
            console.error('Failed to load history:', e);
            return [];
        }
    }

    saveHistory() {
        try {
            localStorage.setItem('developerHistory', JSON.stringify(this.history));
        } catch (e) {
            console.error('Failed to save history:', e);
        }
    }

    highlightCode(element) {
        // Simple syntax highlighting
        const code = element.textContent;
        
        // Python highlighting
        const pythonKeywords = /\b(def|class|if|else|elif|for|while|try|except|finally|return|import|from|as|with|in|is|and|or|not)\b/g;
        const pythonStrings = /(['"])(.*?)\1/g;
        const pythonComments = /#.*$/gm;
        const pythonNumbers = /\b\d+(\.\d+)?\b/g;
        
        let highlighted = code
            .replace(pythonKeywords, '<span class="code-keyword">$&</span>')
            .replace(pythonStrings, '<span class="code-string">$&</span>')
            .replace(pythonComments, '<span class="code-comment">$&</span>')
            .replace(pythonNumbers, '<span class="code-number">$&</span>')
            .replace(/\b(self|cls)\b/g, '<span class="code-parameter">$&</span>')
            .replace(/\b([A-Z][a-zA-Z0-9_]*)\b/g, '<span class="code-class">$&</span>');
        
        element.innerHTML = highlighted;
    }

    copyFixedCode() {
        const code = document.getElementById('fixedCode')?.textContent;
        if (code) {
            Utils.copyToClipboard(code);
        }
    }
}

// Global functions for HTML onclick handlers
function analyzeCode() {
    if (!window.developerAnalyzer) return;
    window.developerAnalyzer.analyzeCode();
}

function getAIHelp() {
    if (!window.developerAnalyzer) return;
    window.developerAnalyzer.getAIHelp();
}

function clearCode() {
    if (!window.developerAnalyzer) return;
    window.developerAnalyzer.clearCode();
}

function loadExample() {
    if (!window.developerAnalyzer) return;
    window.developerAnalyzer.loadExample();
}

function copyFixedCode() {
    if (!window.developerAnalyzer) return;
    window.developerAnalyzer.copyFixedCode();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.developerAnalyzer = new DeveloperAnalyzer();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl + Enter to analyze
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            analyzeCode();
        }
        
        // Ctrl + Shift + F for AI fix
        if (e.ctrlKey && e.shiftKey && e.key === 'F') {
            e.preventDefault();
            getAIHelp();
        }
    });
    
    // Auto-analyze after 2 seconds of inactivity
    const codeInput = document.getElementById('codeInput');
    if (codeInput) {
        codeInput.addEventListener('input', Utils.debounce(() => {
            // Only auto-analyze if code is substantial
            if (codeInput.value.length > 50) {
                window.developerAnalyzer.updateCodeMetrics();
            }
        }, 2000));
    }
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DeveloperAnalyzer;
}