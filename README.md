Check out the app - https://financialagentpranjalrmcf7.streamlit.app/

# Agentic AI Application for Financial Analysis

This project is an advanced **Agentic AI application** that brings together fundamental and technical financial analysis. It leverages **Phidata** for managing infrastructure and deployment, **Groq** for AI capabilities, and integrates interactive visualizations and chatbot functionalities for an enhanced financial decision-making experience.

## Features

### **1. Fundamental Analysis**
- Market Capitalization
- Price-to-Earnings (P/E) Ratio
- Earnings Per Share (EPS)
- Revenue (TTM)
- Gross Profit (TTM)
- Net Income (TTM)
- Operating Cash Flow (TTM)
- Dividend Yield
- 52-Week High/Low
- Return on Assets (ROA) and Return on Equity (ROE)
- Debt-to-Equity Ratio

### **2. Technical Analysis**
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands (Upper and Lower)
- Average True Range (ATR)
- Stochastic Oscillator (%K)
- Simple Moving Averages (SMA) for 50 and 200 periods
- Exponential Moving Averages (EMA)

### **3. Interactive Visualizations**
- Candlestick Charts for price analysis
- Volume Charts for trading activity
- Moving averages and Bollinger Bands overlays

### **4. Chatbot Integration**
- An AI-powered chatbot for personalized financial advice
- Responds to queries like "Should I buy MSFT stock?"
- Provides actionable insights and explains financial metrics

### **5. Backend & Infrastructure**
- **Phidata** for managing infrastructure and enabling seamless integration of AI agents.
- **Groq AI** for handling multi-agent architecture and advanced AI computations.

## Tech Stack
- **Frontend:** Streamlit for creating a responsive and interactive web app.
- **Backend:** Phidata for infrastructure management.
- **AI:** Groq for implementing Agentic AI architecture.
- **Visualization:** Plotly for interactive financial charts.
- **Data:** YFinance for fetching real-time financial data.

## Setup Instructions

### Prerequisites
1. Python 3.8+
2. API Key for **Groq** (Add to `.env` file as `GROQ_API_KEY`)
3. Required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## Usage
- Use the search bar to enter stock tickers or company names.
- Navigate between tabs for **Fundamental Analysis**, **Technical Analysis**, and **Charts**.
- Interact with the chatbot in the sidebar for financial advice.
- Visualize real-time market data with detailed charts and overlays.

## Future Enhancements
- Add multi-language support for global users.
- Incorporate more financial metrics and ratios.
- Enable advanced AI-driven portfolio recommendations.
