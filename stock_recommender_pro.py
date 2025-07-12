
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, AwesomeOscillatorIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator, IchimokuIndicator, PSARIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

default_stocks = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN", "AMD", "NFLX", "IBM"]

def process_stock(ticker):
    df = yf.download(ticker, start="2021-01-01", end="2024-12-31", interval="1d")
    df.dropna(inplace=True)

    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd_diff()
    df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['BB_bbm'] = BollingerBands(close=df['Close']).bollinger_mavg()
    df['BB_bbh'] = BollingerBands(close=df['Close']).bollinger_hband()
    df['BB_bbl'] = BollingerBands(close=df['Close']).bollinger_lband()
    df['STOCH'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    df['ADX'] = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
    df['CCI'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
    df['WilliamsR'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close']).williams_r()
    df['Momentum'] = AwesomeOscillatorIndicator(high=df['High'], low=df['Low']).awesome_oscillator()
    df['Ichimoku_a'] = IchimokuIndicator(df['High'], df['Low']).ichimoku_a()
    df['Ichimoku_b'] = IchimokuIndicator(df['High'], df['Low']).ichimoku_b()
    df['psar'] = PSARIndicator(df['High'], df['Low'], df['Close']).psar()

    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    df.dropna(inplace=True)

    features = [
        'Close', 'RSI', 'MACD', 'EMA20', 'BB_bbm', 'BB_bbh', 'BB_bbl',
        'STOCH', 'ATR', 'ADX', 'CCI', 'WilliamsR', 'Momentum',
        'Ichimoku_a', 'Ichimoku_b', 'psar', 'Volume'
    ]

    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    latest_row = df.iloc[-1]
    latest_features = X.iloc[-1:]
    prediction = model.predict(latest_features)[0]

    if prediction == 1:
        entry_price = latest_row['Close']
        try:
            future_data = df.iloc[-5:]
            exit_price = future_data['High'].mean()
        except:
            exit_price = None

        stop_loss_price = entry_price - latest_row['ATR'] if latest_row['ATR'] else entry_price * 0.97

        if exit_price:
            expected_profit = exit_price - entry_price
            expected_return_pct = (expected_profit / entry_price) * 100
            risk = entry_price - stop_loss_price
            reward_risk_ratio = expected_profit / risk if risk != 0 else None
        else:
            expected_profit = expected_return_pct = reward_risk_ratio = None

        return {
            "ticker": ticker,
            "recommendation": "شراء ✅",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2) if exit_price else "غير متاح",
            "stop_loss": round(stop_loss_price, 2),
            "expected_profit": round(expected_profit, 2) if expected_profit else "غير متاح",
            "expected_return_pct": f"{round(expected_return_pct, 2)}%" if expected_return_pct else "غير متاح",
            "reward_risk_ratio": round(reward_risk_ratio, 2) if reward_risk_ratio else "غير متاح"
        }
    else:
        return {
            "ticker": ticker,
            "recommendation": "بيع ❌",
            "entry_price": "-",
            "exit_price": "-",
            "stop_loss": "-",
            "expected_profit": "-",
            "expected_return_pct": "-",
            "reward_risk_ratio": "-"
        }

st.title("📊 توصيات الأسهم الأمريكية - النسخة الاحترافية بالذكاء الاصطناعي")

st.write("أدخل رموز الأسهم الأمريكية التي ترغب بتحليلها (افصل بينها بفاصلة):")

user_input = st.text_input("الأسهم:", value=",".join(default_stocks))
tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]

if st.button("احصل على التوصيات"):
    if not tickers:
        st.error("يرجى إدخال رمز سهم واحد على الأقل.")
    else:
        with st.spinner("جاري تحليل الأسهم باستخدام مؤشرات احترافية..."):
            results = []
            for ticker in tickers:
                try:
                    res = process_stock(ticker)
                    results.append(res)
                except Exception as e:
                    st.warning(f"حدث خطأ في تحليل {ticker}: {e}")

        for r in results:
            st.markdown(f"### {r['ticker']} - {r['recommendation']}")
            st.write(f"**سعر الدخول:** {r['entry_price']}")
            st.write(f"**سعر الخروج المتوقع:** {r['exit_price']}")
            st.write(f"**وقف الخسارة (ATR):** {r['stop_loss']}")
            st.write(f"**الربح المتوقع:** {r['expected_profit']}")
            st.write(f"**العائد المتوقع (%):** {r['expected_return_pct']}")
            st.write(f"**نسبة العائد إلى المخاطرة:** {r['reward_risk_ratio']}")
            st.write("---")
