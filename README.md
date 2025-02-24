# Portfolio Optimizer App  
**ポートフォリオ最適化アプリ** 

## Description / 説明  
A web application for portfolio optimization with efficient frontier visualization, Monte Carlo simulations, and key performance metrics like expected return, risk, and Sharpe ratio. Users can input stock tickers, set timeframes, and optimize portfolios for specific target returns with downloadable allocations.  

効率的フロンティアの可視化、モンテカルロシミュレーション、期待リターン、リスク、シャープレシオなどのパフォーマンス指標を提供するポートフォリオ最適化のWebアプリです。ユーザーは銘柄を入力し、期間を設定し、特定のターゲットリターンに最適なポートフォリオを作成し、その配分をダウンロードできます。

## Key Features / 主な機能  

### Asset Clustering / 資産クラスタリング  
- **English:** Groups assets into clusters using K-Means based on historical returns.  
- **日本語:** 過去のリターンに基づき、K-Meansクラスタリングで資産をグループ化。

### Historical Performance of Assets / 資産の過去パフォーマンス  
- **English:** Displays cumulative historical performance using time-series plots.  
- **日本語:** 時系列プロットで各資産の累積パフォーマンスを表示。

### Portfolio Optimization / ポートフォリオ最適化  
- **English:** Uses mean-variance optimization to minimize risk for a given target return. Generates an Efficient Frontier.  
- **日本語:** 平均分散最適化を使用し、指定したターゲットリターンに対するリスクを最小化。効率的フロンティアを生成。

### Monte Carlo Simulations / モンテカルロシミュレーション  
- **English:** Simulates thousands of random portfolios to visualize risk-return profiles and Sharpe ratios.  
- **日本語:** 無数のランダムなポートフォリオをシミュレートし、リスク・リターンプロファイルとシャープレシオを可視化。

### Denoising Returns / ノイズ除去リターン  
- **English:** Uses Principal Component Analysis (PCA) to enhance portfolio robustness.  
- **日本語:** 主成分分析（PCA）を使用して、より安定したポートフォリオを構築。

### Portfolio Metrics / ポートフォリオ指標  
- **English:** Calculates and displays key metrics:  
  - Expected Annual Return  
  - Annual Volatility  
  - Sharpe Ratio  
- **日本語:** 主要な指標を計算・表示：  
  - 期待年間リターン  
  - 年間ボラティリティ・リスク  
  - シャープレシオ

### Efficient Frontier Visualization / 効率的フロンティアの可視化  
- **English:** Dynamically visualizes the Efficient Frontier and Monte Carlo results using Plotly.  
- **日本語:** Plotlyを使用し、動的にフロンティアとモンテカルロ結果を可視化。

### Risk Contribution Analysis / リスク貢献度分析  
- **English:** Shows each asset’s contribution to overall portfolio risk.  
- **日本語:** 各資産のポートフォリオ全体のリスクへの影響を表示。

### Sector Allocation / セクター配分  
- **English:** Fetches and visualizes sector/industry allocation of selected stocks.  
- **日本語:** 選択した銘柄のセクター・業界配分を取得・表示。

### Custom Dataset Upload / カスタムデータセットのアップロード  
- **English:** Users can upload CSV files with historical returns for analysis.  
- **日本語:** CSV形式の過去リターンデータをアップロードして分析可能。

### Backtesting Portfolio Performance / ポートフォリオのバックテスト  
- **English:** Simulates historical portfolio performance over time.  
- **日本語:** 過去データを用いてポートフォリオのパフォーマンスをシミュレーション。

### Rebalancing Recommendations / リバランス推奨  
- **English:** Suggests rebalancing strategies based on asset price changes.  
- **日本語:** 資産価格の変動に基づき、リバランス戦略を提案。

### Sensitivity Analysis / 感度分析  
- **English:** Examines how risk-free rate variations impact performance.  
- **日本語:** リスクフリーレートの変動がパフォーマンスに与える影響を分析。

### Portfolio Diversification Score / ポートフォリオ分散スコア  
- **English:** Calculates and displays a score based on asset correlation.  
- **日本語:** 資産間の相関性に基づいた分散スコアを計算・表示。

### Dynamic Visualizations / 動的な可視化  
- **English:** Uses Plotly for interactive scatter plots and comparisons.  
- **日本語:** Plotlyを活用し、インタラクティブな散布図や比較を提供。

### Alerts and Notifications / アラートと通知  
- **English:** Provides notifications for key portfolio insights:  
  - High Sharpe Ratio (indicating strong performance)  
  - Low Sharpe Ratio (suggesting need for adjustment)  
- **日本語:** 主要なポートフォリオインサイトを通知：  
  - 高シャープレシオ – 優れたパフォーマンスの兆候  
  - 低シャープレシオ – 調整の必要性を示唆

### Downloadable Portfolio Allocation / ポートフォリオ配分のダウンロード  
- **English:** Allows users to download optimized allocations in CSV format.  
- **日本語:** 最適化された配分をCSV形式でダウンロード可能。

### User-Friendly Input Options / ユーザーフレンドリーな入力オプション  
- **English:** Provides easy input fields for:  
  - Stock tickers (e.g., AAPL, TSLA, MSFT)  
  - Custom dataset uploads  
  - Target return slider  
  - Risk-free rate adjustments  
- **日本語:** 直感的な入力機能を提供：  
  - 銘柄（例: AAPL, TSLA, MSFT）  
  - カスタムデータセットのアップロード  
  - ターゲットリターンスライダー  
  - リスクフリーレート調整

## Why This Project Matters / このプロジェクトの重要性  
**English:**  
This project demonstrates my expertise in financial engineering, data analysis, and software development. It integrates machine learning, quantitative finance, and web development, showcasing my ability to build data-driven financial applications with Python, Pandas, NumPy, Plotly, and Flask/Django.

**日本語:**  
このプロジェクトは、金融工学、データ分析、ソフトウェア開発のスキルを示しています。機械学習、定量金融、Web開発を統合し、Python、Pandas、NumPy、Plotly、Flask/Django を活用してデータ駆動型の金融アプリケーションを構築する能力を示します。

## How to Use / 使い方  

### Clone the repository / リポジトリをクローン  
```bash
git clone https://github.com/yourusername/portfolio_optimizer_app.git
cd portfolio_optimizer_app
