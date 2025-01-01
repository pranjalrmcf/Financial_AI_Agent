Check out the app - https://financialagentpranjalrmcf7.streamlit.app/

Stock Market AI Assistant 📈
A powerful Streamlit-based web application that combines real-time stock market analysis with AI-powered insights. This tool integrates technical analysis, fundamental metrics, and interactive AI chat capabilities to provide comprehensive stock market intelligence.
Features
Stock Analysis

Real-time Stock Data: Fetch and display current market data for any publicly traded stock
Technical Analysis:

Price charts with candlestick patterns
Volume analysis
Multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
Moving averages (20-day and 50-day SMA)



Fundamental Analysis

Market capitalization
P/E ratio
Revenue metrics
Cash flow analysis
Dividend information
EPS data

AI-Powered Insights

Multi-agent system utilizing:

Web Search Agent for market news and trends
Financial AI Agent for stock-specific analysis
Comprehensive AI Agent combining multiple data sources


Interactive chat interface for custom queries
Historical chat tracking

Prerequisites

Python 3.8+
Groq API key

Installation

Clone the repository:

bashCopygit clone [repository-url]
cd stock-market-ai-assistant

Install required dependencies:

bashCopypip install -r requirements.txt

Create a .env file in the root directory and add your Groq API key:

CopyGROQ_API_KEY=your_api_key_here
Required Dependencies

streamlit
plotly
phi
yfinance
pandas
python-dotenv

Usage

Start the Streamlit application:

bashCopystreamlit run app.py

Access the web interface through your browser (typically at http://localhost:8501)
Enter a stock ticker symbol in the sidebar
Select your preferred time period for analysis
Use the AI chatbot to ask questions about the stock or market conditions

Features Breakdown
Charts and Visualization

Interactive candlestick charts
Volume analysis
Moving averages overlay
Technical indicator visualization

Technical Indicators

Relative Strength Index (RSI)
Average True Range (ATR)
Bollinger Bands
Simple Moving Averages (SMA)
Exponential Moving Averages (EMA)
Stochastic Oscillator

AI Capabilities

Natural language processing for stock-related queries
Real-time web search integration
Financial data analysis
Combined insights from multiple AI agents

Environment Variables
Required environment variables:

GROQ_API_KEY: Your Groq API key for AI functionality

Contributing

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Powered by Groq's LLM models
Built with Streamlit
Uses yfinance for market data
Integrates phi for AI agent functionality
