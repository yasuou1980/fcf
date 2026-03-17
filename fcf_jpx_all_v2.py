# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yfinance",
#     "pandas",
#     "numpy",
#     "xlrd",
#     "requests",
# ]
# ///

import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import warnings
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# Yahoo Financeの401エラーを回避するためのセッション設定
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
})

def get_tse_universe_tickers():
    """JPX公式から東証全銘柄を取得"""
    print("JPX公式から東証全銘柄リストを取得中...", flush=True)
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    df = pd.read_excel(url)
    target_markets = ['プライム', 'スタンダード', 'グロース']
    df = df[df['市場・商品区分'].str.contains('|'.join(target_markets), na=False)]
    
    tickers_info = []
    for _, row in df.iterrows():
        code_raw = str(row['コード']).strip()
        if code_raw.endswith('.0'): code_raw = code_raw[:-2]
        tickers_info.append({
            'Ticker': code_raw + '.T',
            'Company_Name': row['銘柄名']
        })
    print(f"リスト取得完了！対象: {len(tickers_info)}銘柄", flush=True)
    return tickers_info

def fetch_and_calculate_factors(ticker_dict):
    ticker_symbol = ticker_dict['Ticker']
    company_name = ticker_dict['Company_Name']
    
    data = {
        'Ticker': ticker_symbol, 'Company_Name': company_name,
        'Market_Cap': np.nan, 'Enterprise_Value': np.nan,
        'BM_Ratio': np.nan, 'FCF_Yield': np.nan,
        'ROA': np.nan, 'EBITDA_Margin': np.nan,
        'Asset_Growth': np.nan, 'EBITDA_Growth': np.nan,
        'Inv_Dummy': np.nan, 'Price_Range': np.nan,
        'Data_Quality_Flag': ""
    }
    
    try:
        # sessionを渡して接続を安定させる
        ticker = yf.Ticker(ticker_symbol, session=session)
        
        # 安定している fast_info を使用
        fast = ticker.fast_info
        market_cap = fast.get('market_cap')
        data['Market_Cap'] = market_cap
        
        curr = fast.get('last_price')
        h52 = fast.get('year_high')
        l52 = fast.get('year_low')
        if curr and h52 and l52 and (h52 - l52) > 0:
            data['Price_Range'] = (curr - l52) / (h52 - l52)
            
        # 財務データ取得 (401エラーが出やすいため、個別tryで囲む)
        try:
            bs = ticker.balance_sheet
            financials = ticker.financials
            cf = ticker.cashflow
            
            if not bs.empty and not financials.empty:
                # B/M計算
                equity = None
                for col in ['Stockholders Equity', 'Total Stockholder Equity']:
                    if col in bs.index:
                        equity = bs.loc[col].iloc[0]
                        break
                if equity and market_cap:
                    data['BM_Ratio'] = equity / market_cap
                
                # FCF Yield計算
                if not cf.empty and 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
                    fcf = cf.loc['Operating Cash Flow'].iloc[0] + cf.loc['Capital Expenditure'].iloc[0]
                    if market_cap: data['FCF_Yield'] = fcf / market_cap
                
                # ROA計算
                if 'Net Income' in financials.index and 'Total Assets' in bs.index:
                    data['ROA'] = financials.loc['Net Income'].iloc[0] / bs.loc['Total Assets'].iloc[0]

        except:
            pass # 財務データが取れなくても基本データは残す

    except Exception:
        data['Data_Quality_Flag'] = "Fetch Error"
        
    return data

def calculate_scores(df):
    """取得データからスコア算出"""
    print("スコアリング処理を実行中...", flush=True)
    # スコア計算に必要な最低限の列がある銘柄のみ残す
    df = df.dropna(subset=['Market_Cap', 'Price_Range']).copy()
    if df.empty: return df

    # 0-100に正規化 (小さいほど良いものは 100 - rank)
    df['Score_Size'] = 100 - (df['Market_Cap'].rank(pct=True) * 100)
    df['Score_FCFP'] = df['FCF_Yield'].rank(pct=True, na_option='bottom') * 100
    df['Score_BM'] = df['BM_Ratio'].rank(pct=True, na_option='bottom') * 100
    df['Score_ROA'] = df['ROA'].rank(pct=True, na_option='bottom') * 100
    df['Score_Entry'] = 100 - (df['Price_Range'].rank(pct=True) * 100)
    
    # スコア未算出の項目を50点で補完
    score_cols = ['Score_Size', 'Score_FCFP', 'Score_BM', 'Score_ROA', 'Score_Entry']
    df[score_cols] = df[score_cols].fillna(50)
    
    # 総合スコア計算 (日米共通の簡易版ウェイト)
    df['Total_Score'] = (
        0.20 * df['Score_Size'] +
        0.30 * df['Score_FCFP'] +
        0.20 * df['Score_BM'] +
        0.20 * df['Score_ROA'] +
        0.10 * df['Score_Entry']
    ).round(2)
    
    return df

def run_screener_pipeline():
    tickers_info = get_tse_universe_tickers()
    # デバッグ用に件数を絞りたい場合は以下を有効化
    # tickers_info = tickers_info[:100] 
    
    print(f"\n{len(tickers_info)} 銘柄のデータ取得を開始します...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_and_calculate_factors, t): t for t in tickers_info}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker), 1):
            results.append(future.result())
            if i % 100 == 0: print(f"  ... {i} / {len(tickers_info)} 銘柄完了")

    df_raw = pd.DataFrame(results)
    df_scored = calculate_scores(df_raw)
    
    if df_scored.empty:
        print("\n⚠️ 警告: スクリーニング結果が0件です。")
        return

    df_final = df_scored.sort_values(by="Total_Score", ascending=False)
    csv_filename = 'yfinance_multibagger_scored_v1.csv'
    df_final.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ 完了！ '{csv_filename}' を保存しました。")

if __name__ == "__main__":
    run_screener_pipeline()
