import glob
import io
import os

import streamlit as st
import pandas as pd
import requests

# ページの基本設定
st.set_page_config(
    page_title="Multibagger Screener",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------------------------------------
# データ読み込み関数（キャッシュして高速化）
# ---------------------------------------------------------
@st.cache_data
def load_us_data():
    try:
        df = pd.read_csv('sp600_multibagger_scored_v3_sector.csv')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_jp_data():
    try:
        df = pd.read_csv('yfinance_multibagger_scored_v1.csv')
        if not df.empty:
            # 日本株用の数値フォーマット追加
            if 'Market_Cap' in df.columns:
                df['Market_Cap_Billion_JPY'] = df['Market_Cap'] / 1_000_000_000
            if 'FCF_Yield' in df.columns:
                df['FCF_Yield_Pct_Num'] = df['FCF_Yield'] * 100
            if 'Price_Range' in df.columns:
                df['Price_Range_Pct_Num'] = df['Price_Range'] * 100
            if 'Momentum_6M' in df.columns:
                df['Momentum_6M_Pct_Num'] = df['Momentum_6M'] * 100
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# ---------------------------------------------------------
# 金利環境の取得（FRED: FF金利、1日キャッシュ）
# ---------------------------------------------------------
@st.cache_data(ttl=86400)
def get_fed_rate_trend():
    """直近のFF金利と6ヶ月前を比較し、金利環境（上昇/低下・安定）を返す"""
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        fed = pd.read_csv(io.StringIO(resp.text))
        fed.columns = ['date', 'rate']
        fed['rate'] = pd.to_numeric(fed['rate'], errors='coerce')
        fed = fed.dropna().tail(7)  # 直近7ヶ月分
        if len(fed) < 2:
            return None
        latest = fed['rate'].iloc[-1]
        past = fed['rate'].iloc[0]
        return {'latest': latest, 'past': past, 'rising': latest > past + 0.05,
                'asof': fed['date'].iloc[-1]}
    except Exception:
        return None

df_us = load_us_data()
df_jp = load_jp_data()

# ---------------------------------------------------------
# メイン画面ヘッダー
# ---------------------------------------------------------
st.title("🚀 The Alchemy of Multibagger Stocks")
st.markdown("論文のロジックに完全準拠した **日米マルチバガースクリーナー**")

# 金利環境バナー（論文6.5節: Fed利上げ局面は翌年リターンを約8〜12pt押し下げる）
fed_trend = get_fed_rate_trend()
if fed_trend is None:
    st.info("ℹ️ 金利データ（FRED）を取得できませんでした。金利環境の表示をスキップします。")
elif fed_trend['rising']:
    st.warning(
        f"⚠️ **金利上昇局面** — FF金利は直近6ヶ月で {fed_trend['past']:.2f}% → {fed_trend['latest']:.2f}% に上昇"
        f"（{fed_trend['asof']} 時点）。論文の推定では、利上げ局面はマルチバガー型銘柄の翌年リターンを"
        "**約8〜12ポイント押し下げる**環境です。スコアとは別に留意してください。"
    )
else:
    st.success(
        f"✅ **金利安定・低下局面** — FF金利は直近6ヶ月で {fed_trend['past']:.2f}% → {fed_trend['latest']:.2f}%"
        f"（{fed_trend['asof']} 時点）。論文上、グロース系銘柄に追い風の金利環境です。"
    )

st.markdown("---")

# タブの作成
tab_us, tab_jp, tab_review = st.tabs(["🇺🇸 米国小型株 (S&P 600)", "🇯🇵 日本株 (東証全銘柄)", "📈 振り返り検証"])

# =========================================================
# 🇺🇸 米国株タブ
# =========================================================
with tab_us:
    if df_us.empty:
        st.warning("米国株のデータが見つかりません。先に `fcf_sp600_v3.py` を実行してCSVを作成してください。")
    else:
        st.subheader("🇺🇸 S&P 600 マルチバガー候補")
        
        # フィルターUIを横並びで配置
        col1, col2, col3 = st.columns(3)
        with col1:
            all_sectors = sorted(df_us['Sector'].dropna().unique())
            us_sectors = st.multiselect("セクターで絞り込み (US)", options=all_sectors, default=all_sectors)
        with col2:
            us_min_score = st.slider("最低総合スコア (US)", min_value=0, max_value=100, value=70, step=1)
        with col3:
            us_hide_flags = st.checkbox("「要確認(異常値)」フラグを隠す (US)", value=True)
            
        # フィルタリング適用
        filtered_us = df_us[df_us['Sector'].isin(us_sectors)]
        filtered_us = filtered_us[filtered_us['Total_Score'] >= us_min_score]
        if us_hide_flags:
            filtered_us = filtered_us[~filtered_us['Data_Quality_Flag'].astype(str).str.contains("要確認", na=False)]
            
        st.caption(f"抽出結果: {len(filtered_us)} 銘柄")
        
        # 表示設定
        display_cols_us = [
            'Ticker', 'Company_Name', 'Sector', 'Total_Score',
            'FCF_Yield_Pct', 'Price_Range_Pct', 'Momentum_6M_Pct', 'Market_Cap_Billion', 'Data_Quality_Flag'
        ]
        display_cols_us = [c for c in display_cols_us if c in filtered_us.columns]

        st.dataframe(
            filtered_us[display_cols_us],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total_Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "FCF_Yield_Pct": st.column_config.TextColumn("FCF利回り"),
                "Price_Range_Pct": st.column_config.TextColumn("52週安値圏"),
                "Momentum_6M_Pct": st.column_config.TextColumn("6ヶ月リターン"),
                "Market_Cap_Billion": st.column_config.TextColumn("時価総額 ($B)"),
                "Data_Quality_Flag": st.column_config.TextColumn("ステータス")
            }
        )

# =========================================================
# 🇯🇵 日本株タブ
# =========================================================
with tab_jp:
    if df_jp.empty:
        st.warning("日本株のデータが見つかりません。先に `fcf_jpx_all_v2.py` を実行してCSVを作成してください。")
    else:
        st.subheader("🗻 東証全銘柄 マルチバガー候補")
        
        # フィルターUIを横並びで配置
        col1, col2, col3 = st.columns(3)
        with col1:
            jp_max_cap = st.number_input("時価総額の上限 (十億円)", min_value=1, max_value=10000, value=100, step=10)
        with col2:
            jp_min_score = st.slider("最低総合スコア (JP)", min_value=0, max_value=100, value=75, step=1)
        with col3:
            jp_hide_flags = st.checkbox("「要確認(異常値)」フラグを隠す (JP)", value=True)
            
        # フィルタリング適用
        filtered_jp = df_jp[df_jp['Total_Score'] >= jp_min_score]
        filtered_jp = filtered_jp[filtered_jp['Market_Cap_Billion_JPY'] <= jp_max_cap]
        if jp_hide_flags:
            filtered_jp = filtered_jp[~filtered_jp['Data_Quality_Flag'].astype(str).str.contains("要確認", na=False)]
            
        st.caption(f"抽出結果: {len(filtered_jp)} 銘柄")
        
        # 表示設定
        display_cols_jp = [
            'Ticker', 'Company_Name', 'Total_Score',
            'FCF_Yield_Pct_Num', 'Price_Range_Pct_Num', 'Momentum_6M_Pct_Num', 'Market_Cap_Billion_JPY', 'Data_Quality_Flag'
        ]
        display_cols_jp = [c for c in display_cols_jp if c in filtered_jp.columns]

        st.dataframe(
            filtered_jp[display_cols_jp],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("証券コード"),
                "Company_Name": st.column_config.TextColumn("銘柄名"),
                "Total_Score": st.column_config.NumberColumn("総合スコア", format="%.1f"),
                "FCF_Yield_Pct_Num": st.column_config.NumberColumn("FCF利回り (%)", format="%.2f"),
                "Price_Range_Pct_Num": st.column_config.NumberColumn("52週安値圏 (%)", format="%.2f"),
                "Momentum_6M_Pct_Num": st.column_config.NumberColumn("6ヶ月リターン (%)", format="%.2f"),
                "Market_Cap_Billion_JPY": st.column_config.NumberColumn("時価総額 (十億円)", format="%.1f"),
                "Data_Quality_Flag": st.column_config.TextColumn("ステータス")
            }
        )
# =========================================================
# 📈 振り返り検証タブ
# =========================================================

@st.cache_data(ttl=3600)
def load_snapshot(path):
    return pd.read_csv(path)

@st.cache_data(ttl=3600)
def fetch_returns_since(tickers, start_date):
    """スナップショット日以降の各銘柄リターン(%)を取得"""
    import yfinance as yf
    prices = yf.download(tickers, start=start_date, progress=False, auto_adjust=True)['Close']
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.dropna(how='all')
    if prices.empty:
        return pd.Series(dtype=float)
    first = prices.apply(lambda s: s.dropna().iloc[0] if s.dropna().size else float('nan'))
    last = prices.apply(lambda s: s.dropna().iloc[-1] if s.dropna().size else float('nan'))
    return ((last - first) / first * 100).astype(float)

with tab_review:
    st.subheader("📈 スナップショットの振り返り検証")
    st.markdown(
        "毎週保存されたスクリーニング結果（`history/` フォルダ）の上位銘柄が、"
        "その後どれだけのリターンを上げたかをベンチマークと比較します。"
    )

    snapshot_files = sorted(glob.glob('history/*.csv'))
    if not snapshot_files:
        st.info(
            "スナップショットがまだありません。スクリーナーを実行すると `history/` に"
            "日付付きCSVが保存されます（GitHub Actionsで毎週土曜に自動実行されます）。"
        )
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_file = st.selectbox(
                "スナップショットを選択",
                options=snapshot_files,
                format_func=os.path.basename,
                index=0
            )
        with col2:
            top_n = st.slider("検証する上位銘柄数", min_value=5, max_value=50, value=10, step=5)
        with col3:
            review_hide_flags = st.checkbox("「要確認」フラグ銘柄を除外", value=True)

        snap = load_snapshot(selected_file)
        market = 'us' if os.path.basename(selected_file).startswith('us_') else 'jp'
        run_date = str(snap['Run_Date'].iloc[0]) if 'Run_Date' in snap.columns else None

        if run_date is None:
            st.error("このスナップショットには Run_Date 列がありません。")
        else:
            if review_hide_flags and 'Data_Quality_Flag' in snap.columns:
                snap = snap[~snap['Data_Quality_Flag'].astype(str).str.contains("要確認", na=False)]
            top = snap.sort_values('Total_Score', ascending=False).head(top_n).copy()
            tickers = top['Ticker'].astype(str).tolist()
            benchmark = '^GSPC' if market == 'us' else '^N225'
            bench_label = 'S&P 500' if market == 'us' else '日経225'

            with st.spinner(f"{run_date} 以降の株価を取得中..."):
                try:
                    returns = fetch_returns_since(tuple(tickers), run_date)
                    bench_ret = fetch_returns_since((benchmark,), run_date)
                except Exception as e:
                    returns, bench_ret = pd.Series(dtype=float), pd.Series(dtype=float)
                    st.error(f"株価の取得に失敗しました: {e}")

            if returns.empty:
                st.warning("リターンを計算できる株価データがありませんでした。")
            else:
                top['Return_Pct'] = top['Ticker'].astype(str).map(returns)
                avg_ret = top['Return_Pct'].mean()
                bench_val = bench_ret.iloc[0] if not bench_ret.empty else None
                win_rate = (top['Return_Pct'] > 0).mean() * 100

                m1, m2, m3 = st.columns(3)
                m1.metric(f"上位{len(top)}銘柄 平均リターン", f"{avg_ret:+.2f}%")
                if bench_val is not None:
                    m2.metric(
                        f"{bench_label}（同期間）", f"{bench_val:+.2f}%",
                        delta=f"超過 {avg_ret - bench_val:+.2f}pt"
                    )
                m3.metric("勝率（プラス銘柄比率）", f"{win_rate:.0f}%")

                review_cols = [c for c in ['Ticker', 'Company_Name', 'Sector', 'Total_Score', 'Return_Pct'] if c in top.columns]
                st.dataframe(
                    top[review_cols].sort_values('Return_Pct', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Total_Score": st.column_config.NumberColumn("スコア (当時)", format="%.1f"),
                        "Return_Pct": st.column_config.NumberColumn(f"{run_date} 以降リターン (%)", format="%+.2f"),
                    }
                )
                st.caption(
                    "注: 配当・取引コストは未考慮。スコア上位銘柄を均等加重した場合の単純比較です。"
                    "観測期間が短いうちは結果のブレが大きい点に留意してください。"
                )
