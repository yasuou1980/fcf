# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yfinance",
#     "pandas",
#     "numpy",
#     "lxml",
#     "html5lib",
#     "beautifulsoup4",
# ]
# ///

import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. ユニバース取得（米国小型株 S&P 600）
# ==========================================
def get_sp600_universe_tickers():
    print("S&P 600 (米国小型株) の銘柄リストをWikipediaから取得中...", flush=True)
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
    
    tables = pd.read_html(
        url, 
        match='Symbol',
        storage_options={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    
    df = tables[0]
    
    symbol_col = 'Symbol'
    company_col = 'Security' if 'Security' in df.columns else 'Company' if 'Company' in df.columns else df.columns[1]
    sector_col = 'GICS Sector' if 'GICS Sector' in df.columns else 'Sector' if 'Sector' in df.columns else df.columns[2]
    
    tickers = df[symbol_col].str.replace('.', '-').tolist()
    company_names = df[company_col].tolist()
    sectors = df[sector_col].tolist()
    
    tickers_info = []
    for t, n, s in zip(tickers, company_names, sectors):
        tickers_info.append({
            'Ticker': t,
            'Company_Name': n,
            'Sector': s  # セクター情報を追加保持
        })
    
    print(f"リスト取得完了！対象: {len(tickers_info)}銘柄", flush=True)
    return tickers_info

# ==========================================
# 2. 個別銘柄のデータ取得と因子計算
# ==========================================
def fetch_and_calculate_factors(ticker_dict):
    ticker_symbol = ticker_dict['Ticker']
    company_name = ticker_dict['Company_Name']
    sector = ticker_dict['Sector']
    
    data = {
        'Ticker': ticker_symbol,
        'Company_Name': company_name,
        'Sector': sector,
        'Market_Cap': np.nan, 'Enterprise_Value': np.nan,
        'BM_Ratio': np.nan, 'FCF_Yield': np.nan,
        'ROA': np.nan, 'EBITDA_Margin': np.nan,
        'Asset_Growth': np.nan, 'EBITDA_Growth': np.nan,
        'Inv_Dummy': np.nan, 'Price_Range': np.nan,
        'Data_Quality_Flag': ""
    }
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        market_cap = info.get('marketCap')
        data['Market_Cap'] = market_cap
        data['Enterprise_Value'] = info.get('enterpriseValue')
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        high_52w = info.get('fiftyTwoWeekHigh')
        low_52w = info.get('fiftyTwoWeekLow')
        
        if not (high_52w and low_52w) and current_price:
            hist = ticker.history(period="1y")
            if not hist.empty:
                high_52w = hist['High'].max()
                low_52w = hist['Low'].min()
                
        if current_price and high_52w and low_52w and (high_52w - low_52w) > 0:
            data['Price_Range'] = (current_price - low_52w) / (high_52w - low_52w)
            
        bs = ticker.balance_sheet
        financials = ticker.financials
        cf = ticker.cashflow
        
        if not bs.empty and not financials.empty:
            try:
                # Value: B/M
                equity = None
                for col in ['Stockholders Equity', 'Total Stockholder Equity', 'Total Equity Gross Minority Interest']:
                    if col in bs.index:
                        equity = bs.loc[col].iloc[0]
                        break
                if equity and market_cap and market_cap > 0:
                    data['BM_Ratio'] = equity / market_cap
                
                # Value: FCF/P (金融以外で主に意味を持つ)
                fcf = info.get('freeCashflow')
                if fcf is None and not cf.empty:
                    if 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
                        fcf = cf.loc['Operating Cash Flow'].iloc[0] + cf.loc['Capital Expenditure'].iloc[0]
                if fcf and market_cap and market_cap > 0:
                    data['FCF_Yield'] = fcf / market_cap
                    
                # Profitability: ROA
                if 'Net Income' in financials.index and 'Total Assets' in bs.index:
                    net_income = financials.loc['Net Income'].iloc[0]
                    total_assets = bs.loc['Total Assets'].iloc[0]
                    if total_assets and total_assets > 0:
                        data['ROA'] = net_income / total_assets
                
                # Profitability: EBITDA Margin
                ebitda_cols = ['EBITDA', 'Normalized EBITDA']
                rev_cols = ['Total Revenue', 'Revenue']
                ebitda_val = next((financials.loc[c].iloc[0] for c in ebitda_cols if c in financials.index), None)
                rev_val = next((financials.loc[c].iloc[0] for c in rev_cols if c in financials.index), None)
                if pd.notna(ebitda_val) and pd.notna(rev_val) and rev_val > 0:
                    data['EBITDA_Margin'] = ebitda_val / rev_val
                    
                # Investment Pattern
                if len(bs.columns) >= 2 and len(financials.columns) >= 2:
                    assets_y0 = bs.loc['Total Assets'].iloc[0]
                    assets_y1 = bs.loc['Total Assets'].iloc[1]
                    if pd.notna(assets_y0) and pd.notna(assets_y1) and assets_y1 != 0:
                        data['Asset_Growth'] = (assets_y0 - assets_y1) / assets_y1
                        
                    ebitda_y0 = ebitda_val
                    ebitda_y1 = next((financials.loc[c].iloc[1] for c in ebitda_cols if c in financials.index), None)
                    if pd.notna(ebitda_y0) and pd.notna(ebitda_y1) and ebitda_y1 != 0:
                        data['EBITDA_Growth'] = (ebitda_y0 - ebitda_y1) / abs(ebitda_y1)
                        
                    if pd.notna(data['Asset_Growth']) and pd.notna(data['EBITDA_Growth']):
                        data['Inv_Dummy'] = 1 if data['Asset_Growth'] > data['EBITDA_Growth'] else 0
                        
            except Exception:
                pass
                
    except Exception as e:
        data['Data_Quality_Flag'] = "API Fetch Error"
        
    return data

# ==========================================
# 3. セクター別スコアリング処理
# ==========================================
def calculate_scores(df):
    print("データ品質の確認とセクター別スコアリング処理を実行中...", flush=True)
    df = df.dropna(subset=['Market_Cap', 'Price_Range']).copy()
    
    # 異常値フラグ (金融・不動産セクター以外でFCF異常値をチェック)
    fin_real_estate_mask = df['Sector'].isin(['Financials', 'Real Estate'])
    
    df.loc[(~fin_real_estate_mask) & (df['FCF_Yield'] > 0.30), 'Data_Quality_Flag'] = "要確認: FCF異常値(>30%)"
    df.loc[(~fin_real_estate_mask) & (df['FCF_Yield'] < -0.50), 'Data_Quality_Flag'] = "要確認: 大幅なFCF赤字"
    df.loc[fin_real_estate_mask, 'Data_Quality_Flag'] = "金融・不動産セクター特化評価"
    
    # --- 共通スコア計算 (0〜100点に正規化) ---
    df['Size_Proxy'] = df['Enterprise_Value'].fillna(df['Market_Cap'])
    df['Score_Size'] = 100 - (df['Size_Proxy'].rank(pct=True) * 100)
    
    df['Score_BM'] = df['BM_Ratio'].rank(pct=True) * 100
    df['Score_FCFP'] = df['FCF_Yield'].rank(pct=True) * 100
    df['Score_ROA'] = df['ROA'].rank(pct=True) * 100
    df['Score_EBITDA_Margin'] = df['EBITDA_Margin'].rank(pct=True) * 100
    
    df['Score_Profitability'] = pd.concat([df['Score_ROA'], df['Score_EBITDA_Margin']], axis=1).mean(axis=1)
    
    conditions = [
        (df['Asset_Growth'] > 0) & (df['Inv_Dummy'] == 0),
        (df['Inv_Dummy'] == 1)
    ]
    df['Score_Investment'] = np.select(conditions, [100, 0], default=50)
    df['Score_Entry'] = 100 - (df['Price_Range'].rank(pct=True) * 100)
    
    # 欠損値フォールバック
    score_cols = ['Score_Size', 'Score_BM', 'Score_FCFP', 'Score_ROA', 'Score_Profitability', 'Score_Investment', 'Score_Entry']
    df[score_cols] = df[score_cols].fillna(50)
    
    # --- セクター別の Total Score 計算 ---
    df['Total_Score'] = 0.0
    
    # パターンA: 一般セクター (論文標準モデル)
    mask_gen = ~fin_real_estate_mask
    df.loc[mask_gen, 'Total_Score'] = (
        0.15 * df.loc[mask_gen, 'Score_Size'] +
        0.20 * df.loc[mask_gen, 'Score_BM'] +
        0.25 * df.loc[mask_gen, 'Score_FCFP'] +
        0.15 * df.loc[mask_gen, 'Score_Profitability'] +
        0.15 * df.loc[mask_gen, 'Score_Investment'] +
        0.10 * df.loc[mask_gen, 'Score_Entry']
    )
    
    # パターンB: 金融・不動産セクター (FCF・EBITDAを除外、B/MとROAを極大化)
    # ウェイト: BM(40%), ROA(35%), Size(15%), Entry(10%)
    df.loc[fin_real_estate_mask, 'Total_Score'] = (
        0.15 * df.loc[fin_real_estate_mask, 'Score_Size'] +
        0.40 * df.loc[fin_real_estate_mask, 'Score_BM'] +
        0.35 * df.loc[fin_real_estate_mask, 'Score_ROA'] +
        0.10 * df.loc[fin_real_estate_mask, 'Score_Entry']
    )
    
    df['Total_Score'] = df['Total_Score'].round(2)
    for col in score_cols:
        df[col] = df[col].round(1)
        
    df['FCF_Yield_Pct'] = (df['FCF_Yield'] * 100).round(2).astype(str) + '%'
    df['Price_Range_Pct'] = (df['Price_Range'] * 100).round(2).astype(str) + '%'
    df['Market_Cap_Billion'] = (df['Market_Cap'] / 1_000_000_000).round(2).astype(str) + 'B'
        
    return df

# ==========================================
# 4. メイン実行パイプライン
# ==========================================
def run_screener_pipeline():
    tickers_info = get_sp600_universe_tickers()
    
    print(f"\n全 {len(tickers_info)} 銘柄の米国小型株データ取得・計算を開始します...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(fetch_and_calculate_factors, t): t for t in tickers_info}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker), 1):
            results.append(future.result())
            if i % 50 == 0:
                print(f"  ... {i} / {len(tickers_info)} 銘柄完了", flush=True)

    df_raw = pd.DataFrame(results)
    df_scored = calculate_scores(df_raw)
    df_sorted = df_scored.sort_values(by="Total_Score", ascending=False)
    
    output_cols = [
        'Ticker', 'Company_Name', 'Sector', 'Total_Score', 'Score_Size', 'Score_FCFP', 
        'Market_Cap_Billion', 'Enterprise_Value', 'BM_Ratio', 'FCF_Yield_Pct', 
        'ROA', 'EBITDA_Margin', 'Asset_Growth', 'EBITDA_Growth', 
        'Inv_Dummy', 'Price_Range_Pct', 'Data_Quality_Flag'
    ]
    df_final = df_sorted[output_cols].copy()
    
    csv_filename = 'sp600_multibagger_scored_v3_sector.csv'
    df_final.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    print("\n✅ セクター別スコアリング＆スクリーニングが完了しました！")
    print(f"結果を '{csv_filename}' に保存しました。")
    print("\n=== 総合スコア トップ10銘柄 (米国小型株) ===")
    
    display_cols = ['Ticker', 'Sector', 'Total_Score', 'FCF_Yield_Pct', 'Price_Range_Pct', 'Data_Quality_Flag']
    print(df_final[display_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    run_screener_pipeline()