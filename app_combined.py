import streamlit as st
import pandas as pd

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
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df_us = load_us_data()
df_jp = load_jp_data()

# ---------------------------------------------------------
# メイン画面ヘッダー
# ---------------------------------------------------------
st.title("🚀 The Alchemy of Multibagger Stocks")
st.markdown("論文のロジックに完全準拠した **日米マルチバガースクリーナー**")
st.markdown("---")

# タブの作成
tab_us, tab_jp = st.tabs(["🇺🇸 米国小型株 (S&P 600)", "🇯🇵 日本株 (東証全銘柄)"])

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
            'FCF_Yield_Pct', 'Price_Range_Pct', 'Market_Cap_Billion', 'Data_Quality_Flag'
        ]
        
        st.dataframe(
            filtered_us[display_cols_us],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total_Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "FCF_Yield_Pct": st.column_config.TextColumn("FCF利回り"),
                "Price_Range_Pct": st.column_config.TextColumn("52週安値圏"),
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
            'FCF_Yield_Pct_Num', 'Price_Range_Pct_Num', 'Market_Cap_Billion_JPY', 'Data_Quality_Flag'
        ]
        
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
                "Market_Cap_Billion_JPY": st.column_config.NumberColumn("時価総額 (十億円)", format="%.1f"),
                "Data_Quality_Flag": st.column_config.TextColumn("ステータス")
            }
        )