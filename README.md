# Deep Research Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![DeepSeek](https://img.shields.io/badge/Model-DeepSeek%20R1-green.svg)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/shiv207/Deepsearch-clone-opensource/releases)

**Tags:** `#ai-research` `#deep-learning` `#nlp` `#semantic-search` `#groq` `#deepseek` `#serpapi` `#python` `#research-tool` `#information-retrieval` `#web-scraping` `#content-analysis` `#machine-learning` `#text-analysis` `#open-source`

## Overview
This project is an experimental attempt to replicate deep search capabilities similar to frontier models like Grok-3 and OpenAI's models, using the DeepSeek R1 model via Groq's API. The results have been surprisingly positive, demonstrating that even with simpler models, we can achieve meaningful research capabilities.

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git
- A Groq API key (get it from [Groq's website](https://groq.com))
- A SerpAPI key (get it from [SerpAPI's website](https://serpapi.com))

### Step 1: Clone the Repository
```bash
git clone https://github.com/shiv207/Deepsearch-clone-opensource.git
cd Deepsearch-clone-opensource
```

### Step 2: Create and Activate Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory with your API keys:
```plaintext
GROQ_API_KEY=your_groq_api_key_here
SERPAPI_KEY=your_serpapi_key_here
```

### Step 5: Verify Installation
```bash
python main.py
```
When prompted, enter a research query to test the system.

### Troubleshooting
- If you encounter SSL certificate errors, ensure your Python installation has proper SSL certificates.
- If you get dependency conflicts, try creating a fresh virtual environment.
- For NLTK-related errors, run:
  ```bash
  python -c "import nltk; nltk.download('punkt')"
  ```

### Note on API Keys
- Never commit your `.env` file to version control
- Keep your API keys secure and don't share them
- Monitor your API usage to avoid unexpected charges

## Key Features
- Uses DeepSeek R1 Distill Qwen 32B model through Groq's API
- Integrates with SerpAPI for comprehensive web search capabilities
- Implements sophisticated content validation and scoring
- Provides research paper-level analysis and formatting
- Includes multi-source verification and cross-referencing

## Performance
While not reaching the capabilities of frontier models like Grok-3, the system has performed better than expected for a simple project. The combination of:
- DeepSeek R1 for analysis
- SerpAPI for reliable search results
- Custom content validation
- Multi-source verification

has resulted in a robust research tool that can provide meaningful insights and analysis.

## Architecture
The system uses a multi-stage approach:
1. Multi-source search using SerpAPI
2. Content validation and scoring
3. Deep analysis using DeepSeek R1 via Groq
4. Research paper-style formatting and presentation

## Dependencies
- Groq API for DeepSeek R1 model access
- SerpAPI for web search capabilities
- Various Python libraries for content processing and analysis

## Note
This is an experimental project aimed at exploring the possibilities of creating research capabilities with more accessible models. While it doesn't match the capabilities of frontier models, it demonstrates that meaningful research automation is possible with current, publicly available tools.

## Results
The system has shown promising results in:
- Comprehensive information gathering
- Source validation
- Content analysis
- Research paper-style presentation

While there's still a significant gap between this and frontier models, the results are encouraging for a proof-of-concept project.

## Future Work
This project serves as a foundation for exploring how smaller, more accessible models can be leveraged for research tasks. Future improvements could focus on:
- Enhanced source validation
- Better content synthesis
- More sophisticated analysis techniques
- Integration with additional data sources

## Disclaimer
This is an experimental project and should not be considered a replacement for professional research tools or frontier models. It serves as a proof of concept for what can be achieved with publicly available models and APIs. 