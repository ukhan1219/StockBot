StockBot
StockBot is a personal project aimed at developing a robust and adaptable tool for analyzing and predicting market movements, specifically in the futures commodities market. This project leverages data from two different APIs—TradingView and yfinance—to retrieve real-time and historical data for various commodities across multiple time frames.

Project Overview
As part of my ongoing learning journey, I am exploring and testing various machine learning and artificial intelligence techniques to identify the best model(s) for StockBot. My focus includes:

Markov Chain Monte Carlo (MCMC): A method used for sampling from probability distributions based on constructing a Markov chain.
Metropolis-Hastings: A specific algorithm within the MCMC framework that I am currently learning to implement successfully and fine-tune for predicting stock prices.
The code in this repository is constantly evolving as I experiment with different models and sampling techniques to optimize the bot's performance.

APIs Used
TradingView API: Used for real-time data retrieval and chart analysis. This API provides the ability to track current futures commodities and is integral to the webhook scripts.
yfinance API: Utilized for retrieving historical data and ensuring the robustness of the model by testing across different time frames.
Webhook Integration
The project also includes two webhook scripts designed to interact with a sister script on TradingView's platform. These scripts will send alerts for buy and sell signals, which will be visualized by placing tags directly on the current TradingView chart. This feature aims to provide real-time actionable insights based on the data analysis performed by StockBot.

Learning and Development
Please note that this project is a work in progress. I am continuously learning about and integrating new machine learning models, sampling techniques, and AI methodologies. As I refine my approach and improve the bot, you may see frequent updates and changes to the codebase.

Getting Started
To get started with StockBot, you will need to:

Clone this repository.
Set up API keys for TradingView and yfinance.
Run the provided scripts to start data retrieval and analysis.
Monitor the output and webhook alerts for buy/sell signals.
Contributing
Since this project is part of my learning process, I am open to suggestions, ideas, and contributions. Feel free to open an issue or submit a pull request if you have any improvements or would like to collaborate.
