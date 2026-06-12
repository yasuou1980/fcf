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
import time

warnings.filterwarnings("ignore", category=FutureWarning)

# ブラウザ偽装用セッション
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
})

def get_tse_universe_tickers():
    """JPX公式から東証全銘柄リストを取得"""
    print("JPX公式から東証全銘柄リストを取得中...", flush=True)
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        df = pd.read_excel(url)
    except:
        # 万が一URLが変わった場合の予備
        return []
        
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
        'Market_Cap': np.nan, 'Price_Range': np.nan,
        'BM_Ratio': np.nan, 'FCF_Yield': np.nan, 'ROA': np.nan,
        'EBITDA_Margin': np.nan, 'Asset_Growth': np.nan,
        'EBITDA_Growth': np.nan, 'Inv_Dummy': np.nan,
        'Momentum_6M': np.nan,
        'Data_Quality_Flag': ""
    }
    
    # 複数回リトライ
    for attempt in range(2):
        try:
            ticker = yf.Ticker(ticker_symbol, session=session)
            fast = ticker.fast_info
            
            # 基本データの抽出
            m_cap = fast.get('market_cap')
            curr = fast.get('last_price')
            h52 = fast.get('year_high')
            l52 = fast.get('year_low')
            
            if m_cap and curr and h52 and l52:
                data['Market_Cap'] = m_cap
                if (h52 - l52) > 0:
                    data['Price_Range'] = (curr - l52) / (h52 - l52)
                
                # 財務データ取得の試行
                try:
                    # BS/CF等は失敗しやすいため個別で管理
                    bs = ticker.balance_sheet
                    cf = ticker.cashflow
                    fin = ticker.financials
                    
                    if not bs.empty:
                        equity = None
                        for col in ['Stockholders Equity', 'Total Stockholder Equity']:
                            if col in bs.index:
                                equity = bs.loc[col].iloc[0]
                                break
                        if equity: data['BM_Ratio'] = equity / m_cap
                        
                    if not cf.empty and 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
                        fcf = cf.loc['Operating Cash Flow'].iloc[0] + cf.loc['Capital Expenditure'].iloc[0]
                        data['FCF_Yield'] = fcf / m_cap

                    if not fin.empty and 'Net Income' in fin.index and not bs.empty and 'Total Assets' in bs.index:
                        data['ROA'] = fin.loc['Net Income'].iloc[0] / bs.loc['Total Assets'].iloc[0]

                    # Profitability: EBITDA Margin
                    ebitda_cols = ['EBITDA', 'Normalized EBITDA']
                    rev_cols = ['Total Revenue', 'Revenue']
                    ebitda_val = rev_val = None
                    if not fin.empty:
                        ebitda_val = next((fin.loc[c].iloc[0] for c in ebitda_cols if c in fin.index), None)
                        rev_val = next((fin.loc[c].iloc[0] for c in rev_cols if c in fin.index), None)
                    if pd.notna(ebitda_val) and pd.notna(rev_val) and rev_val > 0:
                        data['EBITDA_Margin'] = ebitda_val / rev_val

                    # Investment Pattern (資産成長 vs EBITDA成長)
                    if not bs.empty and not fin.empty and len(bs.columns) >= 2 and len(fin.columns) >= 2:
                        if 'Total Assets' in bs.index:
                            assets_y0 = bs.loc['Total Assets'].iloc[0]
                            assets_y1 = bs.loc['Total Assets'].iloc[1]
                            if pd.notna(assets_y0) and pd.notna(assets_y1) and assets_y1 != 0:
                                data['Asset_Growth'] = (assets_y0 - assets_y1) / assets_y1
                        ebitda_y1 = next((fin.loc[c].iloc[1] for c in ebitda_cols if c in fin.index), None)
                        if pd.notna(ebitda_val) and pd.notna(ebitda_y1) and ebitda_y1 != 0:
                            data['EBITDA_Growth'] = (ebitda_val - ebitda_y1) / abs(ebitda_y1)
                        if pd.notna(data['Asset_Growth']) and pd.notna(data['EBITDA_Growth']):
                            data['Inv_Dummy'] = 1 if data['Asset_Growth'] > data['EBITDA_Growth'] else 0
                except:
                    pass # 財務データが取れなくても基本データがあれば続行

                # Technical: 6ヶ月モメンタム（論文では負の係数 = 直近下落銘柄が有利）
                try:
                    hist_6m = ticker.history(period="7mo")
                    if not hist_6m.empty and len(hist_6m) > 20:
                        first_close = hist_6m['Close'].iloc[0]
                        if first_close and first_close > 0:
                            data['Momentum_6M'] = (hist_6m['Close'].iloc[-1] - first_close) / first_close
                except:
                    pass


                break # 取得成功したらループを抜ける
            
        except:
            time.sleep(1) # 失敗したら少し待つ
            
    return data

def calculate_scores(df):
    """取得データからスコア算出"""
    print("スコアリング処理を実行中...", flush=True)
    # 最低限必要な時価総額または価格のどちらかがあれば残す（厳格すぎると0件になるため）
    df = df.dropna(subset=['Market_Cap']).copy()
    if df.empty: return df

    # 異常値フラグ (米国版と同基準)
    df.loc[df['FCF_Yield'] > 0.30, 'Data_Quality_Flag'] = "要確認: FCF異常値(>30%)"
    df.loc[df['FCF_Yield'] < -0.50, 'Data_Quality_Flag'] = "要確認: 大幅なFCF赤字"

    # 正規化（欠損値は平均的な50点とする）
    df['Score_Size'] = 100 - (df['Market_Cap'].rank(pct=True, na_option='keep') * 100)
    df['Score_FCFP'] = df['FCF_Yield'].rank(pct=True, na_option='keep') * 100
    df['Score_BM'] = df['BM_Ratio'].rank(pct=True, na_option='keep') * 100
    df['Score_ROA'] = df['ROA'].rank(pct=True, na_option='keep') * 100
    df['Score_EBITDA_Margin'] = df['EBITDA_Margin'].rank(pct=True, na_option='keep') * 100
    df['Score_Profitability'] = pd.concat([df['Score_ROA'], df['Score_EBITDA_Margin']], axis=1).mean(axis=1)

    # Investment Pattern: 資産成長がEBITDA成長の範囲内なら満点、超過(Inv_Dummy=1)なら0点
    conditions = [
        (df['Asset_Growth'] > 0) & (df['Inv_Dummy'] == 0),
        (df['Inv_Dummy'] == 1)
    ]
    df['Score_Investment'] = np.select(conditions, [100, 0], default=50)

    # 価格圏が取れない場合は平均値
    if 'Price_Range' in df.columns:
        df['Score_Entry'] = 100 - (df['Price_Range'].rank(pct=True, na_option='keep') * 100)
    else:
        df['Score_Entry'] = 50

    # 論文: 3-6ヶ月モメンタムは負の係数 → 直近6ヶ月の下落が大きいほど高得点
    df['Score_Momentum'] = 100 - (df['Momentum_6M'].rank(pct=True, na_option='keep') * 100)

    score_cols = ['Score_Size', 'Score_FCFP', 'Score_BM', 'Score_ROA',
                  'Score_Profitability', 'Score_Investment', 'Score_Entry', 'Score_Momentum']
    for col in score_cols:
        df[col] = df[col].fillna(50)

    # 米国版（一般セクター）と同一ウェイト
    df['Total_Score'] = (
        0.15 * df['Score_Size'] +
        0.20 * df['Score_BM'] +
        0.25 * df['Score_FCFP'] +
        0.15 * df['Score_Profitability'] +
        0.10 * df['Score_Investment'] +
        0.10 * df['Score_Entry'] +
        0.05 * df['Score_Momentum']
    ).round(2)

    return df

def run_screener_pipeline():
    tickers_info = get_tse_universe_tickers()
    if not tickers_info:
        print("銘柄リストが取得できませんでした。")
        return
        
    print(f"\n{len(tickers_info)} 銘柄のデータ取得を開始します...")
    results = []
    # 重要：スレッド数を10に落としてブロックを回避
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_and_calculate_factors, t): t for t in tickers_info}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker), 1):
            results.append(future.result())
            if i % 100 == 0: print(f"  ... {i} / {len(tickers_info)} 銘柄完了", flush=True)

    df_raw = pd.DataFrame(results)
    df_scored = calculate_scores(df_raw)
    
    if df_scored.empty:
        print("\n⚠️ 警告: スクリーニング結果が0件です。")
        return

    # カラム調整
    df_final = df_scored.sort_values(by="Total_Score", ascending=False)
    csv_filename = 'yfinance_multibagger_scored_v1.csv'
    df_final.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ 完了！結果を '{csv_filename}' に保存しました。")

if __name__ == "__main__":
    run_screener_pipeline()
