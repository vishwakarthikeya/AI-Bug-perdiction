// Student Mode JavaScript
class StudentChat {
    constructor() {
        this.messages = [];
        this.isTyping = false;
        this.apiBaseUrl = 'http://127.0.0.1:8000';
        
        this.init();
    }

    init() {
        // Load initial message
        this.addMessage({
            sender: 'ai',
            content: "Hello! I'm your AI learning assistant. I can help you understand bugs, explain code issues, and teach you programming concepts. Ask me anything or upload code for analysis!",
            timestamp: new Date()
        });

        // Set up event listeners
        this.setupEventListeners();
        
        // Load topics
        this.loadTopics();
    }

    setupEventListeners() {
        // Send button
        document.getElementById('userInput')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // File upload
        document.getElementById('codeFile')?.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        // Example buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const example = btn.getAttribute('onclick').match(/'(\w+)'/)?.[1];
                this.loadExample(example);
            });
        });

        // Topic buttons
        document.querySelectorAll('.topic-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const topic = btn.getAttribute('onclick').match(/'(\w+)'/)?.[1];
                this.loadTopic(topic);
            });
        });
    }

    async sendMessage() {
        const input = document.getElementById('userInput');
        const message = input?.value.trim();
        
        if (!message || this.isTyping) return;

        // Add user message
        this.addMessage({
            sender: 'user',
            content: message,
            timestamp: new Date()
        });

        // Clear input
        if (input) input.value = '';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Send to backend
            const response = await fetch(`${this.apiBaseUrl}/student-chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            // Hide typing indicator
            this.hideTypingIndicator();

            // Add AI response
            this.addMessage({
                sender: 'ai',
                content: data.response || data.explanation || "I apologize, but I couldn't generate a response. Please try again.",
                timestamp: new Date()
            });

        } catch (error) {
            console.error('Error:', error);
            this.hideTypingIndicator();
            
            this.addMessage({
                sender: 'ai',
                content: "I'm having trouble connecting to the server. Please check your connection and try again.",
                timestamp: new Date()
            });
        }
    }

    async handleFileUpload(file) {
        if (!file) return;

        // Show loading
        this.showTypingIndicator();

        const reader = new FileReader();
        reader.onload = async (e) => {
            const code = e.target.result;
            
            // Add user message with code
            this.addMessage({
                sender: 'user',
                content: `Uploaded file: ${file.name}\n\n\`\`\`\n${code}\n\`\`\``,
                timestamp: new Date()
            });

            try {
                // Send to backend
                const response = await fetch(`${this.apiBaseUrl}/student-chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        message: `Please analyze this code and explain any potential bugs:\n\n${code}` 
                    })
                });

                const data = await response.json();
                this.hideTypingIndicator();

                // Add AI response
                this.addMessage({
                    sender: 'ai',
                    content: data.response || data.explanation || "I've analyzed your code. Here's what I found...",
                    timestamp: new Date()
                });

            } catch (error) {
                console.error('Error:', error);
                this.hideTypingIndicator();
                
                this.addMessage({
                    sender: 'ai',
                    content: "Failed to analyze the file. Please try again.",
                    timestamp: new Date()
                });
            }
        };

        reader.readAsText(file);
    }

    addMessage(message) {
        this.messages.push(message);
        this.renderMessages();
        
        // Save to localStorage
        this.saveChatHistory();
    }

    renderMessages() {
        const container = document.getElementById('chatMessages');
        if (!container) return;

        container.innerHTML = '';
        
        this.messages.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.sender}-message`;
            
            const timeStr = msg.timestamp.toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            const senderName = msg.sender === 'ai' ? 'AI Assistant' : 'You';
            const senderIcon = msg.sender === 'ai' ? 'fas fa-robot' : 'fas fa-user';
            
            // Format code blocks
            let content = msg.content;
            content = this.formatCodeBlocks(content);
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <i class="${senderIcon}"></i> ${senderName}
                </div>
                <div class="message-content">${content}</div>
                <div class="message-time">${timeStr}</div>
            `;
            
            container.appendChild(messageDiv);
        });

        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    }

    formatCodeBlocks(text) {
        // Convert markdown code blocks to styled HTML
        return text
            .replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                return `<pre><code class="language-${lang || 'text'}">${this.escapeHtml(code)}</code></pre>`;
            })
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showTypingIndicator() {
        this.isTyping = true;
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = 'flex';
            
            // Scroll to bottom
            const container = document.getElementById('chatMessages');
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        }
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
    }

    loadTopic(topic) {
        const topics = {
            memory_leak: "Memory leaks occur when a program fails to release memory that is no longer needed. This can happen in languages without automatic garbage collection (like C/C++) or when circular references prevent garbage collection.",
            null_pointer: "A null pointer exception occurs when you try to access or modify an object reference that points to null (nothing). Always check for null before accessing object properties or methods.",
            infinite_loop: "Infinite loops happen when the loop condition never becomes false. This can be caused by incorrect loop conditions, missing increment statements, or logic errors.",
            buffer_overflow: "Buffer overflow occurs when data written to a buffer exceeds its allocated size, overwriting adjacent memory. This is a serious security vulnerability common in C/C++ programs.",
            race_condition: "Race conditions happen in concurrent programming when multiple threads access shared data simultaneously without proper synchronization, leading to unpredictable results."
        };

        if (topics[topic]) {
            this.addMessage({
                sender: 'ai',
                content: topics[topic],
                timestamp: new Date()
            });
        }
    }

    loadExample(example) {
        const examples = {
            example1: {
                code: `// Memory leak example in C++
void createLeak() {
    int* ptr = new int[100];  // Allocate memory
    // Forgot to delete[] ptr;  // Memory leak!
}`,
                question: "Why does this code cause a memory leak?"
            },
            example2: {
                code: `// Null pointer example
public class Example {
    public static void main(String[] args) {
        String str = null;
        System.out.println(str.length());  // NullPointerException!
    }
}`,
                question: "How to prevent null pointer exceptions?"
            },
            example3: {
                code: `// Infinite loop example
while (true) {
    System.out.println("This will run forever!");
    // Missing break or condition change
}`,
                question: "What's wrong with this loop and how to fix it?"
            }
        };

        if (examples[example]) {
            const { code, question } = examples[example];
            
            // Add user message with question
            this.addMessage({
                sender: 'user',
                content: `${question}\n\n\`\`\`\n${code}\n\`\`\``,
                timestamp: new Date()
            });

            // Simulate AI response after delay
            setTimeout(() => {
                this.showTypingIndicator();
                setTimeout(() => {
                    this.hideTypingIndicator();
                    
                    const responses = {
                        example1: "This code causes a memory leak because memory is allocated with `new int[100]` but never freed with `delete[] ptr`. In C++, you must manually manage heap memory. The fix is to add `delete[] ptr;` before the function returns.",
                        example2: "This causes a NullPointerException because `str` is null and we're trying to call `length()` on it. Always check for null: `if (str != null) System.out.println(str.length());` or use Optional in modern Java.",
                        example3: "This is an infinite loop because the condition is always true. To fix it, add a break condition or make the condition depend on a variable that changes inside the loop."
                    };

                    this.addMessage({
                        sender: 'ai',
                        content: responses[example] || "Let me analyze this example...",
                        timestamp: new Date()
                    });
                }, 1500);
            }, 500);
        }
    }

    clearChat() {
        if (confirm("Clear all chat messages?")) {
            this.messages = [];
            this.renderMessages();
            localStorage.removeItem('studentChatHistory');
        }
    }

    saveChatHistory() {
        try {
            const history = this.messages.map(msg => ({
                ...msg,
                timestamp: msg.timestamp.getTime()  // Convert to serializable format
            }));
            localStorage.setItem('studentChatHistory', JSON.stringify(history));
        } catch (e) {
            console.error('Failed to save chat history:', e);
        }
    }

    loadChatHistory() {
        try {
            const saved = localStorage.getItem('studentChatHistory');
            if (saved) {
                const history = JSON.parse(saved);
                this.messages = history.map(msg => ({
                    ...msg,
                    timestamp: new Date(msg.timestamp)
                }));
                this.renderMessages();
            }
        } catch (e) {
            console.error('Failed to load chat history:', e);
        }
    }

    formatCode() {
        const input = document.getElementById('userInput');
        if (!input) return;

        const code = input.value;
        // Simple code formatting - in a real app, use a proper formatter
        const formatted = code
            .replace(/\t/g, '  ')  // Convert tabs to spaces
            .replace(/\n{3,}/g, '\n\n')  // Limit consecutive newlines
            .trim();

        input.value = formatted;
        Utils.showNotification('Code formatted!', 'success');
    }
}

// Global functions for HTML onclick handlers
function sendMessage() {
    if (!window.studentChat) return;
    window.studentChat.sendMessage();
}

function clearChat() {
    if (!window.studentChat) return;
    window.studentChat.clearChat();
}

function formatCode() {
    if (!window.studentChat) return;
    window.studentChat.formatCode();
}

function uploadCodeFile() {
    const fileInput = document.getElementById('codeFile');
    if (fileInput && window.studentChat) {
        if (fileInput.files.length > 0) {
            window.studentChat.handleFileUpload(fileInput.files[0]);
        } else {
            fileInput.click();
        }
    }
}

function loadTopic(topic) {
    if (!window.studentChat) return;
    window.studentChat.loadTopic(topic);
}

function loadExample(example) {
    if (!window.studentChat) return;
    window.studentChat.loadExample(example);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.studentChat = new StudentChat();
    
    // Add syntax highlighting
    if (typeof hljs !== 'undefined') {
        hljs.highlightAll();
    }
    
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl + / to clear chat
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            clearChat();
        }
        
        // Ctrl + F to format code
        if (e.ctrlKey && e.key === 'f') {
            e.preventDefault();
            formatCode();
        }
    });
    
    // Load saved chat history
    setTimeout(() => {
        window.studentChat.loadChatHistory();
    }, 100);
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StudentChat;
}