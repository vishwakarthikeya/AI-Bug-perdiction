# 🐛 AI Bug Predictor

![AI Bug Predictor](https://img.shields.io/badge/AI-Bug_Predictor-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-lightblue)
![Machine Learning](https://img.shields.io/badge/ML-Logistic_Regression-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

**AI Bug Predictor** is a hackathon-ready, fully functional system that uses Machine Learning and AI to predict, prioritize, and fix bugs in source code. The system provides:

- 🎓 **Student Mode**: AI-powered explanations for learning programming concepts
- 👨‍💻 **Developer Mode**: ML-based bug prediction with severity classification and AI fixes
- 🤖 **AI Integration**: Gemini API for explanations + DeepSeek API for code fixes
- 📊 **Visual Analytics**: Interactive charts and metrics for code analysis

## 🚀 Features

### 🎓 Student Mode
- Interactive chatbot for learning about bugs
- AI-powered explanations with examples
- Code upload and analysis
- Learning resources and examples

### 👨‍💻 Developer Mode
- Real-time bug probability prediction
- Severity classification (Critical/High/Medium/Low)
- AI-generated code fixes
- Code metrics and visualizations
- History tracking

### 🏗️ Technical Features
- **ML Model**: Logistic Regression trained on 50k samples
- **AI Integration**: Gemini + DeepSeek APIs with fallback responses
- **Modern Stack**: FastAPI backend + Three.js frontend
- **Responsive Design**: Works on all devices
- **Dark Theme**: Neon aesthetics with smooth animations

## 🏆 Hackathon Relevance

This project is perfect for hackathons because:

✅ **COMPLETE SOLUTION**: End-to-end working application  
✅ **AI/ML INTEGRATION**: Combines ML prediction with AI explanations  
✅ **EDUCATIONAL VALUE**: Helps both students and developers  
✅ **PRODUCTION-READY**: Clean architecture with proper error handling  
✅ **VISUALLY APPEALING**: Modern UI with animations and charts  
✅ **EASY TO RUN**: Simple setup with clear instructions  
✅ **SCALABLE**: Modular design for future enhancements  

## 🛠️ Tech Stack

### Frontend
- HTML5, CSS3 (with animations and gradients)
- JavaScript (ES6+)
- Three.js (3D background)
- Chart.js (visualizations)

### Backend
- Python 3.9+
- FastAPI (async web framework)
- Scikit-learn (machine learning)
- Pydantic (data validation)

### Machine Learning
- Logistic Regression model
- Feature extraction from code
- Probability prediction
- Severity classification

### AI Services
- Google Gemini API (explanations)
- DeepSeek API (code fixes)
- Fallback responses when APIs unavailable

## 📁 Project Structure
AI-Bug-Predictor/
├── frontend/ # Frontend files
│ ├── index.html # Landing page
│ ├── student.html # Student learning interface
│ ├── developer.html # Developer analysis interface
│ ├── css/ # Stylesheets
│ └── js/ # JavaScript files
├── backend/ # Python backend
│ ├── main.py # FastAPI application
│ ├── ml_service.py # ML prediction service
│ ├── ai_service.py # AI integration service
│ ├── schemas.py # Pydantic models
│ ├── utils.py # Utility functions
│ └── requirements.txt # Python dependencies
├── ml/ # Machine learning
│ ├── bug_prediction.ipynb # Jupyter notebook
│ ├── train_model.py # Training script
│ └── model.pkl # Trained model
├── dataset/ # Training data
│ └── bug_dataset_50k.csv # 50k sample dataset
├── docs/ # Documentation
│ ├── architecture.txt # System architecture
│ └── math_explanation.txt # Mathematical foundations
└── README.md # This file
## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Modern web browser (Chrome, Firefox, Edge)
- VS Code (recommended)

### Installation & Running

#### Option 1: VS Code (Recommended)

1. **Clone/Download the project**
   ```bash
   git clone <repository-url>
   cd AI-Bug-Predictor