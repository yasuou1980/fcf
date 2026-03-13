# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yfinance",
#     "pandas",
#     "numpy",
#     "xlrd",
# ]
# ///

import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import warnings

# yfinanceの不要な警告をミュート
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. ユニバース取得（データソース仕様 4.3）
# ==========================================
def get_tse_universe_tickers():
    """JPX公式から東証全銘柄を取得し、ユニバースを作成する"""
    print("JPX公式から東証全銘柄リストを取得中...", flush=True)
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    df = pd.read_excel(url)
    
    # プライム・スタンダード・グロース市場に限定
    target_markets = ['プライム', 'スタンダード', 'グロース']
    df = df[df['市場・商品区分'].str.contains('|'.join(target_markets), na=False)]
    
    # Tickerと企業名の辞書を作成（後で結合するため）
    tickers_info = []
    for _, row in df.iterrows():
        ticker = str(int(float(row['コード']))) + '.T'
        tickers_info.append({
            'Ticker': ticker,
            'Company_Name': row['銘柄名'],
            'Market_Segment': row['市場・商品区分'],
            'Sector': row['33業種区分']
        })
    
    print(f"リスト取得完了！対象: {len(tickers_info)}銘柄", flush=True)
    return tickers_info

# ==========================================
# 2. 個別銘柄のデータ取得と因子計算（指標定義 6）
# ==========================================
def fetch_and_calculate_factors(ticker_dict):
    ticker_symbol = ticker_dict['Ticker']
    company_name = ticker_dict['Company_Name']
    
    # --- 初期値設定 ---
    data = {
        'Ticker': ticker_symbol,
        'Company_Name': company_name,
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
        
        # --- 基本データ ---
        market_cap = info.get('marketCap')
        data['Market_Cap'] = market_cap
        data['Enterprise_Value'] = info.get('enterpriseValue')
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # --- 6.5 Entry因子 (Price Range) ---
        high_52w = info.get('fiftyTwoWeekHigh')
        low_52w = info.get('fiftyTwoWeekLow')
        
        # infoから取れない場合のフォールバック（履歴から計算）
        # ※ current_price の有無に関わらず、52週データがなければ履歴を取得する
        if not (high_52w and low_52w):
            hist = ticker.history(period="1y")
            if not hist.empty:
                high_52w = hist['High'].max()
                low_52w = hist['Low'].min()
                # infoにcurrent_priceもない場合は履歴の最終終値で代替
                if not current_price:
                    current_price = float(hist['Close'].iloc[-1])

        if current_price and high_52w and low_52w and (high_52w - low_52w) > 0:
            data['Price_Range'] = (current_price - low_52w) / (high_52w - low_52w)
            
        # --- 財務データの取得 ---
        bs = ticker.balance_sheet
        financials = ticker.financials
        cf = ticker.cashflow
        
        if not bs.empty and not financials.empty:
            # 最新年のデータ
            try:
                # --- 6.2 Value因子 ---
                # Book-to-Market (自己資本 / 時価総額)
                equity = None
                for col in ['Stockholders Equity', 'Total Stockholder Equity', 'Total Equity Gross Minority Interest']:
                    if col in bs.index:
                        equity = bs.loc[col].iloc[0]
                        break
                if equity and market_cap and market_cap > 0:
                    data['BM_Ratio'] = equity / market_cap
                
                # FCF/P (Free Cash Flow / Market Cap)
                fcf = info.get('freeCashflow')
                if fcf is None and not cf.empty:
                    # FCFがinfoにない場合のフォールバック (営業CF - 設備投資)
                    if 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
                        fcf = cf.loc['Operating Cash Flow'].iloc[0] + cf.loc['Capital Expenditure'].iloc[0] # CapExは通常マイナス
                
                if fcf and market_cap and market_cap > 0:
                    data['FCF_Yield'] = fcf / market_cap
                    
                # --- 6.3 Profitability因子 ---
                # ROA (Net Income / Total Assets)
                if 'Net Income' in financials.index and 'Total Assets' in bs.index:
                    net_income = financials.loc['Net Income'].iloc[0]
                    total_assets = bs.loc['Total Assets'].iloc[0]
                    if total_assets and total_assets > 0:
                        data['ROA'] = net_income / total_assets
                
                # EBITDA Margin
                ebitda_cols = ['EBITDA', 'Normalized EBITDA']
                rev_cols = ['Total Revenue', 'Revenue']
                ebitda_val = next((financials.loc[c].iloc[0] for c in ebitda_cols if c in financials.index), None)
                rev_val = next((financials.loc[c].iloc[0] for c in rev_cols if c in financials.index), None)
                
                if pd.notna(ebitda_val) and pd.notna(rev_val) and rev_val > 0:
                    data['EBITDA_Margin'] = ebitda_val / rev_val
                    
                # --- 6.4 Investment Pattern因子 ---
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
                pass # 内部計算エラーはスキップしてNaNのままにする
                
    except Exception as e:
        data['Data_Quality_Flag'] = "API Fetch Error"
        
    return data

# ==========================================
# 3. スコアリング処理（スクリーニング方式 7.2）
# ==========================================
def calculate_scores(df):
    """取得したデータ群からパーセンタイルを計算し、総合スコアを算出する"""
    print("データ品質の確認とスコアリング処理を実行中...", flush=True)
    
    # --- 8.3 除外条件 ---
    # 時価総額がない、またはPrice Rangeが計算できない銘柄は除外
    df = df.dropna(subset=['Market_Cap', 'Price_Range']).copy()
    
    # --- 異常値フラグ処理 ---
    df.loc[df['FCF_Yield'] > 0.30, 'Data_Quality_Flag'] = "要確認: FCF異常値(>30%)"
    df.loc[df['FCF_Yield'] < -0.50, 'Data_Quality_Flag'] = "要確認: 大幅なFCF赤字"
    
    # --- 個別スコア計算 (0〜100点に正規化) ---
    # rank(pct=True) は 0.0〜1.0 のパーセンタイルを返す。NaNは無視される。
    
    # 1. Size: TEV優先、なければMarket Cap。小さいほど高得点なので 100 - rank
    df['Size_Proxy'] = df['Enterprise_Value'].fillna(df['Market_Cap'])
    df['Score_Size'] = 100 - (df['Size_Proxy'].rank(pct=True) * 100)
    
    # 2. Value: B/M と FCF/P。高いほど高得点
    df['Score_BM'] = df['BM_Ratio'].rank(pct=True) * 100
    df['Score_FCFP'] = df['FCF_Yield'].rank(pct=True) * 100
    
    # 3. Profitability: ROAとEBITDA Margin。高いほど高得点
    roa_score = df['ROA'].rank(pct=True) * 100
    ebitda_score = df['EBITDA_Margin'].rank(pct=True) * 100
    # 両方ある場合は平均、片方ならそちらを採用
    df['Score_Profitability'] = pd.concat([roa_score, ebitda_score], axis=1).mean(axis=1)
    
    # 4. Investment Pattern: 
    # Asset Growth > 0 かつ Inv Dummy == 0 なら高得点(100)、ダミーが1なら低得点(0)、欠損は中立(50)
    conditions = [
        (df['Asset_Growth'] > 0) & (df['Inv_Dummy'] == 0),
        (df['Inv_Dummy'] == 1)
    ]
    choices = [100, 0]
    df['Score_Investment'] = np.select(conditions, choices, default=50)
    
    # 5. Entry: Price Range が低いほど高得点
    df['Score_Entry'] = 100 - (df['Price_Range'].rank(pct=True) * 100)
    
    # --- 8.1 欠損値のフォールバック ---
    # 計算できなかったスコアは中立値（50点）で埋めて総合スコア計算から脱落させない
    score_cols = ['Score_Size', 'Score_BM', 'Score_FCFP', 'Score_Profitability', 'Score_Investment', 'Score_Entry']
    df[score_cols] = df[score_cols].fillna(50)
    
    # --- Total Score の算出 (推奨重み適用) ---
    df['Total_Score'] = (
        0.15 * df['Score_Size'] +
        0.20 * df['Score_BM'] +
        0.25 * df['Score_FCFP'] +
        0.15 * df['Score_Profitability'] +
        0.15 * df['Score_Investment'] +
        0.10 * df['Score_Entry']
    )
    
    # 小数点以下を丸める
    df['Total_Score'] = df['Total_Score'].round(2)
    for col in score_cols:
        df[col] = df[col].round(1)
        
    return df

# ==========================================
# 4. メイン実行パイプライン
# ==========================================
def run_screener_pipeline():
    # テスト時はここを小さく絞ってデバッグ可能です
    tickers_info = get_tse_universe_tickers()
    
    # 高速化のため、最初の数銘柄で試す場合は以下のコメントアウトを外してください
    # tickers_info = tickers_info[:50] 
    
    print(f"\n全 {len(tickers_info)} 銘柄のデータ取得・計算を開始します...")
    print("⚠️ 注意: ネットワーク環境により【30分〜1時間程度】かかります。", flush=True)
    
    results = []
    # 並列処理 (負荷を考慮して20スレッド程度を推奨)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(fetch_and_calculate_factors, t): t for t in tickers_info}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker), 1):
            results.append(future.result())
            if i % 100 == 0:
                print(f"  ... {i} / {len(tickers_info)} 銘柄完了", flush=True)

    df_raw = pd.DataFrame(results)
    
    # スコアリング実行
    df_scored = calculate_scores(df_raw)
    
    # Total Score 降順でソート
    df_sorted = df_scored.sort_values(by="Total_Score", ascending=False)
    
    # 出力列の整理 (出力仕様 9.1)
    output_cols = [
        'Ticker', 'Company_Name', 'Total_Score', 'Score_Size', 'Score_FCFP', 
        'Market_Cap', 'Enterprise_Value', 'BM_Ratio', 'FCF_Yield', 
        'ROA', 'EBITDA_Margin', 'Asset_Growth', 'EBITDA_Growth', 
        'Inv_Dummy', 'Price_Range', 'Data_Quality_Flag'
    ]
    df_final = df_sorted[output_cols].copy()
    
    # CSV出力（空データによる既存ファイルの上書きを防止）
    csv_filename = 'yfinance_multibagger_scored_v1.csv'
    if df_final.empty:
        print("\n⚠️ 警告: スクリーニング結果が0件です。データ取得に問題がある可能性があります。")
        print("既存のCSVを保護するため、上書きをスキップします。")
        return
    df_final.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    print("\n✅ スコアリング＆スクリーニングが完了しました！")
    print(f"結果を '{csv_filename}' に保存しました。")
    print("\n=== 総合スコア トップ10銘柄 ===")
    
    # コンソール表示用に一部の列だけ抽出して表示
    display_cols = ['Ticker', 'Company_Name', 'Total_Score', 'FCF_Yield', 'Price_Range', 'Data_Quality_Flag']
    print(df_final[display_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    run_screener_pipeline()