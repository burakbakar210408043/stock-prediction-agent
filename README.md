# 📈 AI-Powered Stock Prediction Agent

This project is a comprehensive stock market analysis tool that leverages Generative AI to provide real-time data visualization, future price predictions, and expert-level market commentary.

## 🚀 Features

* **Real-Time Data:** Fetches live stock data using `yfinance` (Yahoo Finance).
* **AI Forecasting (Chronos):** Visualizes the potential future trajectory of stock prices for the next 10 days using time-series forecasting models.
* **Smart Analysis (Llama 3 via Groq):** Interprets complex numerical data and technical indicators into easy-to-understand natural language commentary.
* **Interactive Chatbot:** A built-in AI assistant capable of answering specific questions about the analyzed stock.
* **User-Friendly Interface:** Built with Streamlit for a clean and responsive web experience.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Data Source:** Yahoo Finance (`yfinance`)
* **AI Models:**
    * **LLM:** Llama 3 (via Groq API) for text generation and analysis.
    * **Time Series:** Amazon Chronos (via Hugging Face) for chart prediction.

## 📂 Project Structure

* `main.py`: The entry point of the Streamlit application. Handles the UI and user interactions.
* `agent_brain.py`: Contains the core logic for AI models (Llama 3 & Chronos) and data processing.
* `tools/`: Directory containing utility scripts like `data_fetcher.py` for retrieving stock data.
* `Baslat.bat`: A batch script to easily launch the application on Windows.
* `.env`: Configuration file for storing API keys securely.
* `requirements.txt`: List of Python dependencies.

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/burakbakar210408043/stock-prediction-agent.git](https://github.com/burakbakar210408043/stock-prediction-agent.git)
    cd stock-prediction-agent
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys**
    * Create a file named `.env` in the root directory.
    * Add your Groq and Hugging Face API keys:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    HUGGINGFACE_API_KEY=your_huggingface_key_here
    ```

## ▶️ How to Run

You can run the application in two ways:

### Option 1: Using the Batch Script (Recommended for Windows)
Simply double-click the **`Baslat.bat`** file in the project folder. This will automatically open the application in your default web browser.

### Option 2: Using Terminal
Open your terminal or command prompt in the project directory and run:
```bash
streamlit run main.py
Burak Bakar
Emir Satı
Emirhan Daldal
