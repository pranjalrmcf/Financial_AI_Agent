import os
import streamlit as st
import plotly.graph_objects as go
from phi.agent.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['GROQ_API_KEY']

# Streamlit Page Config
st.set_page_config(
    page_title="Stock Market AI Assistant",
    page_icon="📈",
    layout="wide"
)

# Helper Functions
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data dynamically."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval="1d", auto_adjust=True)
        if hist.empty:
            st.error(f"No historical data available for {ticker}.")
            return None, None
        info = stock.info
        return info, hist
    except Exception as e:
        st.error(f"Failed to fetch stock data for {ticker}: {e}")
        return None, None

def plot_price_chart(hist, ticker):
    """Plot stock price chart with moving averages."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name="Price"
    ))
    # Add Moving Averages
    ma20 = hist['Close'].rolling(window=20).mean()
    ma50 = hist['Close'].rolling(window=50).mean()
    fig.add_trace(go.Scatter(x=hist.index, y=ma20, name="20-Day SMA", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=hist.index, y=ma50, name="50-Day SMA", line=dict(color='orange')))
    fig.update_layout(
        title=f"{ticker} Stock Price",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_white"
    )
    return fig

def plot_volume_chart(hist):
    """Plot trading volume chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist.index,
        y=hist['Volume'],
        name="Volume",
        marker_color="blue"
    ))
    fig.update_layout(
        title="Trading Volume",
        yaxis_title="Volume",
        xaxis_title="Date",
        template="plotly_white"
    )
    return fig

def display_technical_indicators(hist):
    """Display technical indicators."""
    st.subheader("Technical Indicators")
    cols = st.columns(3)  # Consistent grid layout

    # RSI, MACD, ATR
    with cols[0]:
        period = 14
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        st.metric("RSI (14)", f"{rsi.iloc[-1]:.2f}")

        atr = hist['High'] - hist['Low']
        st.metric("ATR (14)", f"{atr.rolling(window=period).mean().iloc[-1]:.2f}")

    # Bollinger Bands
    with cols[1]:
        rolling_mean = hist['Close'].rolling(window=20).mean()
        rolling_std = hist['Close'].rolling(window=20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        st.metric("Upper Bollinger Band", f"{upper_band.iloc[-1]:.2f}")
        st.metric("Lower Bollinger Band", f"{lower_band.iloc[-1]:.2f}")

    # SMA, EMA, Stochastic %K
    with cols[2]:
        sma_50 = hist['Close'].rolling(window=50).mean()
        ema_50 = hist['Close'].ewm(span=50, adjust=False).mean()
        st.metric("SMA (50)", f"{sma_50.iloc[-1]:.2f}")
        st.metric("EMA (50)", f"{ema_50.iloc[-1]:.2f}")
        
        high_14 = hist['High'].rolling(window=14).max()
        low_14 = hist['Low'].rolling(window=14).min()
        stoch_k = ((hist['Close'] - low_14) / (high_14 - low_14)) * 100
        st.metric("Stochastic %K", f"{stoch_k.iloc[-1]:.2f}")

def display_fundamental_analysis(info):
    """Display fundamental analysis metrics."""
    st.subheader("Fundamental Analysis")
    cols = st.columns(3)  # Adjust number of columns if needed
    
    with cols[0]:
        st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
        st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
        st.metric("Operating Cash Flow (TTM)", f"${info.get('operatingCashflow', 'N/A'):,}")
    
    with cols[1]:
        st.metric("Revenue (TTM)", f"${info.get('totalRevenue', 'N/A'):,}")
        st.metric("EPS (TTM)", f"{info.get('trailingEps', 'N/A')}")
        st.metric("Net Income (TTM)", f"${info.get('netIncomeToCommon', 'N/A'):,}")
    
    with cols[2]:
        st.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
        st.metric("Dividend Yield", f"{info.get('dividendYield', 'N/A') * 100:.2f}%" if info.get('dividendYield') else "N/A")

# Initialize agents
def initialize_agents():
    """
    Initializes the Streamlit session state agents (web_agent, finance_agent, multi_ai_agent).
    Includes basic debugging prints to confirm that the API key is loaded and to catch any exceptions.
    """
    if not st.session_state.get('agents_initialized', False):
        # Debugging: Verify the API key is loaded.
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            st.error("GROQ_API_KEY is not set. Make sure you have it in your .env file and call load_dotenv().")
            return  # Early exit if we don’t have a valid key

        try:
            # Web Search Agent
            st.session_state['web_agent'] = Agent(
                name="Web Search Agent",
                role="Search the web for information",
                model=Groq(id="llama-3.3-70b-versatile", api_key=api_key),
                tools=[DuckDuckGo(fixed_max_results=5)],
                instructions=["Always include sources"],
                show_tool_calls=True,
                markdown=True
            )
            
            # Financial AI Agent
            st.session_state['finance_agent'] = Agent(
                name="Financial AI Agent",
                role="Provide financial insights",
                model=Groq(id="llama-3.3-70b-versatile", api_key=api_key),
                tools=[YFinanceTools(stock_price=True, company_news=True, analyst_recommendations=True)],
                instructions=["Provide detailed analysis with visualizations"],
                show_tool_calls=True,
                markdown=True
            )
            
            # Comprehensive AI Agent
            st.session_state['multi_ai_agent'] = Agent(
                name="Comprehensive AI Agent",
                role="Combine financial and web insights",
                model=Groq(id="llama-3.3-70b-versatile", api_key=api_key),
                team=[
                    st.session_state['web_agent'], 
                    st.session_state['finance_agent']
                ],
                instructions=["Provide actionable insights and include sources."],
                show_tool_calls=True,
                markdown=True
            )
            
            st.session_state['agents_initialized'] = True
            st.success("Agents have been successfully initialized!")
        
        except Exception as e:
            st.error(f"Failed to initialize agents: {str(e)}")

# Initialize session state attributes
if 'agents_initialized' not in st.session_state:
    st.session_state['agents_initialized'] = False
if 'web_agent' not in st.session_state:
    st.session_state['web_agent'] = None
if 'finance_agent' not in st.session_state:
    st.session_state['finance_agent'] = None
if 'multi_ai_agent' not in st.session_state:
    st.session_state['multi_ai_agent'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


# Streamlit UI
st.title("📈 Stock Market AI Assistant")

# User Input
st.sidebar.header("Stock Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., NVDA):")
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ["1 Month", "3 Months", "6 Months", "1 Year"],
    index=3
)
period_mapping = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}
selected_period = period_mapping[time_period]

# Chatbot Interface
st.sidebar.subheader("💬 Ask the AI Chatbot")
question = st.sidebar.text_area("Ask AI:", placeholder="Type your question here...")

# Buttons for submission and clearing chat
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    ask_button = st.button("Ask AI")
with col2:
    clear_button = st.button("Clear Chat")

# Clear chat history
if clear_button:
    st.session_state['chat_history'] = []
    st.success("Chat history cleared.")

# Ask AI
if ask_button:
    if question.strip():
        # 1. Ensure agents are initialized
        if not st.session_state['agents_initialized']:
            initialize_agents()
        
        # 2. Make sure the 'multi_ai_agent' is set
        if not st.session_state.get('multi_ai_agent'):
            st.error("Multi AI Agent not initialized. Please check your API key and agent setup.")
        else:
            try:
                with st.spinner("Processing your query..."):
                    # 3. Run the agent in stream mode
                    response_generator = st.session_state['multi_ai_agent'].run(question, stream=True)
                    
                    # 4. Build the response text incrementally
                    response_text = ""
                    for item in response_generator:
                        # 'item' could be a string or a token object with a .content attribute
                        if hasattr(item, "content"):
                            response_text += item.content
                        else:
                            response_text += str(item)

                    # 5. Store the conversation in session state
                    st.session_state['chat_history'].append({
                        "question": question, 
                        "response": response_text
                    })
                    
                    # 6. Display the final AI response
                    st.markdown("### AI Response")
                    st.markdown(response_text, unsafe_allow_html=True)

            except Exception as e:
                # More detailed error logging
                st.error(f"An error occurred while processing your request:\n{str(e)}")

    else:
        st.warning("Please enter a query.")

# Display chat history
if st.session_state.get('chat_history'):
    st.markdown("### Chat History")
    for chat in st.session_state['chat_history']:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**AI:** {chat['response']}")
        st.markdown("---")

# Main Tabs
if ticker:
    st.markdown("---")
    st.header(f"Analysis for {ticker.upper()}")

    # Fetch data
    info, hist = fetch_stock_data(ticker, selected_period)
    if info and hist is not None:
        tabs = st.tabs(["Fundamental Analysis", "Technical Analysis", "Charts"])
        
        with tabs[0]:  # Fundamental Analysis Tab
            # Fundamental analysis doesn't need a time period
            display_fundamental_analysis(info)
        
        with tabs[1]:  # Technical Analysis Tab
            # Technical analysis uses historical data
            display_technical_indicators(hist)
        
        with tabs[2]:  # Charts Tab
            # Visualizations of historical data
            st.subheader("Price Chart")
            st.plotly_chart(plot_price_chart(hist, ticker), use_container_width=True)
            st.subheader("Volume Chart")
            st.plotly_chart(plot_volume_chart(hist), use_container_width=True)

