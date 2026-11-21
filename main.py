import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os

# ============================================================
# 한글 폰트 설정 (깨짐 방지)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CSV 인코딩 자동 판별 (utf-8-sig -> cp949)
# ============================================================
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ============================================================
# 열 이름 표준화 (학교마다 다른 열 이름 -> 표준 이름)
# ============================================================
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)

    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num

    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ============================================================
# 학교명 추출 (파일명 앞부분)
# ============================================================
def get_school_name(file):
    if file is None:
        return ""
    return os.path.splitext(os.path.basename(file.name))[0]

# ============================================================
# 파일 업로드 (사이드바)
# ============================================================
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 CSV", type="csv")

# ============================================================
# 데이터 로드 (한 번만, @st.cache_data 로 캐싱)
# ============================================================
@st.cache_data
def load_data(file):
    if file is None:
        return None
    with st.spinner("CSV 파일을 읽고 있어요..."):
        return std_cols(read_csv(file))

env_df = load_data(env_file)
growth_df = load_data(growth_file)

# ============================================================
# 사이드바 정보
# ============================================================
school = get_school_name(env_file) if env_file is not None else (get_school_name(growth_file) if growth_file is not None else "")
if school:
    st.sidebar.success(f"학교: {school}")
elif env_df is not None or growth_df is not None:
    st.sidebar.info("파일명은 인식했지만 학교명을 찾지 못했어요.")
else:
    st.sidebar.info("CSV 파일을 업로드해주세요.")

metric_options = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
selected_metrics = st.sidebar.multiselect(
    "생육 측정 항목 선택",
    metric_options,
    default=metric_options,
    help="막대 그래프에 표시할 항목을 선택하세요."
)

# ============================================================
# 그래프 함수 정의
# ============================================================
def render_bar_chart(data, x_col, y_col, title):
    """막대 그래프 함수 (재사용)"""
    if data is None or len(data) == 0:
        st.info("막대 그래프를 그릴 데이터가 없어요.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data[x_col], y=data[y_col], marker_color="skyblue"))
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    st.plotly_chart(fig, width="stretch")

def render_scatter_trendline(x_data, y_data, x_label, y_label, title):
    """산점도 + 추세선 (px.scatter trendline 금지 -> np.polyfit 으로 직접 계산)"""
    mask = x_data.notna() & y_data.notna()
    x_v = x_data[mask].values
    y_v = y_data[mask].values
    if len(x_v) < 2:
        st.info("추세선을 그리려면 데이터가 2개 이상 필요해요.")
        return
    r = np.corrcoef(x_v, y_v)[0, 1]
    slope, intercept = np.polyfit(x_v, y_v, 1)
    x_line = np.linspace(x_v.min(), x_v.max(), 100)
    y_line = slope * x_line + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_v, y=y_v, mode="markers",
        marker=dict(color="steelblue", size=8),
        name="개체 데이터"
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="red", dash="dash"),
        name=f"추세선 (r={r:.3f})"
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    st.plotly_chart(fig, width="stretch")

def render_boxplot(data, col, title):
    """박스플롯 함수"""
    vals = data[col].dropna()
    if len(vals) == 0:
        st.info("박스플롯을 그릴 데이터가 없어요.")
        return
    fig = go.Figure()
    fig.add_trace(go.Box(y=vals, name=title, boxpoints="outliers"))
    fig.update_layout(
        yaxis_title=col,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    st.plotly_chart(fig, width="stretch")

# ============================================================
# 탭 생성
# ============================================================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ============================================================
# 탭 1: 실험 개요
# ============================================================
with tab1:
    try:
        st.subheader("🔬 연구 배경 및 목적")
        st.write("극지식물은 혹독한 환경에 적응한 식물로, 생육 조건의 미세한 차이가 개체별 생장 차이에 영향을 줍니다.")
        st.write("본 대시보드는 우리 학교 극지식물 개체의 생육 데이터와 환경 데이터를 수집하여, 환경 조건과 개체 특성 간의 관계를 시각화하고 식물의 생육 패턴을 분석합니다.")
        st.write("학생 여러분들이 데이터를 직접 탐색하면서 생물 다양성과 환경 요인의 영향을 이해할 수 있도록 구성되었습니다.")

        st.subheader("📋 데이터 요약")
        summary_rows = []
        if growth_df is not None and not growth_df.empty:
            n_individuals = growth_df["개체번호"].nunique()
            summary_rows.append(["총 개체수", n_individuals])
            growth_data_cols = [c for c in growth_df.columns if c not in ("개체번호", "time")]
            summary_rows.append(["측정 항목", ", ".join(growth_data_cols) if growth_data_cols else "없음"])
            if "time" in growth_df.columns:
                date_range = growth_df["time"].dropna()
                if len(date_range) > 0:
                    summary_rows.append(["측정 기간", f"{date_range.min().strftime('%Y-%m-%d')} ~ {date_range.max().strftime('%Y-%m-%d')}"])
        if env_df is not None and not env_df.empty:
            env_data_cols = [c for c in env_df.columns if c not in ("time", "date")]
            summary_rows.append(["환경 측정 항목", ", ".join(env_data_cols) if env_data_cols else "없음"])
            if "time" in env_df.columns:
                date_range = env_df["time"].dropna()
                if len(date_range) > 0:
                    summary_rows.append(["환경 측정 기간", f"{date_range.min().strftime('%Y-%m-%d')} ~ {date_range.max().strftime('%Y-%m-%d')}"])
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows, columns=["항목", "내용"])
            st.dataframe(summary_df, width="stretch")
        else:
            st.info("데이터 요약 정보를 찾지 못했어요.")

        st.subheader("📊 주요 지표")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
                st.metric("평균 생중량 (g)", f"{growth_df['생중량'].mean():.3f}")
            else:
                st.metric("평균 생중량 (g)", "데이터 없음")
        with col2:
            if env_df is not None and not env_df.empty and "temperature" in env_df.columns:
                st.metric("평균 온도 (°C)", f"{env_df['temperature'].mean():.2f}")
            else:
                st.metric("평균 온도 (°C)", "데이터 없음")
        with col3:
            if env_df is not None and not env_df.empty and "humidity" in env_df.columns:
                st.metric("평균 습도 (%)", f"{env_df['humidity'].mean():.2f}")
            else:
                st.metric("평균 습도 (%)", "데이터 없음")
        with col4:
            if growth_df is not None and not growth_df.empty:
                st.metric("총 개체수", growth_df["개체번호"].nunique())
            else:
                st.metric("총 개체수", "데이터 없음")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ============================================================
# 탭 2: 환경 데이터
# ============================================================
with tab2:
    try:
        st.subheader("🌡️ 환경 데이터 요약")
        if env_df is not None and not env_df.empty:
            num_cols = [c for c in env_df.columns if c not in ("time", "date") and pd.api.types.is_numeric_dtype(env_df[c])]
            sum_rows = []
            for col in num_cols:
                vals = env_df[col].dropna()
                if len(vals) > 0:
                    sum_rows.append([col, f"{vals.mean():.2f}", f"{vals.min():.2f}", f"{vals.max():.2f}"])
            if sum_rows:
                sum_df = pd.DataFrame(sum_rows, columns=["항목", "평균", "최소", "최대"])
                st.dataframe(sum_df, width="stretch")
            else:
                st.info("환경 데이터에서 숫자 열을 찾지 못했어요.")

            if "time" in env_df.columns:
                env_plot = env_df.dropna(subset=["time"]).copy()
                env_plot["date"] = env_plot["time"].dt.strftime("%Y-%m-%d")
                fig1 = make_subplots(rows=1, cols=2, subplot_titles=("온도 변화", "습도 변화"))
                if "temperature" in env_plot.columns:
                    daily_temp = env_plot.groupby("date")["temperature"].mean().reset_index()
                    fig1.add_trace(go.Scatter(
                        x=daily_temp["date"], y=daily_temp["temperature"],
                        mode="lines", name="온도", line=dict(color="red")
                    ), row=1, col=1)
                    fig1.update_yaxes(title_text="온도 (°C)", row=1, col=1)
                if "humidity" in env_plot.columns:
                    daily_hum = env_plot.groupby("date")["humidity"].mean().reset_index()
                    fig1.add_trace(go.Scatter(
                        x=daily_hum["date"], y=daily_hum["humidity"],
                        mode="lines", name="습도", line=dict(color="blue")
                    ), row=1, col=2)
                    fig1.update_yaxes(title_text="습도 (%)", row=1, col=2)
                fig1.update_layout(title="날짜별 환경 변화", font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
                st.plotly_chart(fig1, width="stretch")

            if "ph" in env_df.columns or "ec" in env_df.columns:
                st.subheader("📊 pH·EC 변화 (일별)")
                fig2 = make_subplots(rows=1, cols=1)
                if "ph" in env_df.columns:
                    ph_daily = env_df.groupby(env_df["time"].dt.strftime("%Y-%m-%d"))["ph"].mean().reset_index()
                    ph_daily.columns = ["date", "ph"]
                    fig2.add_trace(go.Scatter(
                        x=ph_daily["date"], y=ph_daily["ph"],
                        mode="lines+markers", name="pH", line=dict(color="green")
                    ))
                if "ec" in env_df.columns:
                    ec_daily = env_df.groupby(env_df["time"].dt.strftime("%Y-%m-%d"))["ec"].mean().reset_index()
                    ec_daily.columns = ["date", "ec"]
                    fig2.add_trace(go.Scatter(
                        x=ec_daily["date"], y=ec_daily["ec"],
                        mode="lines+markers", name="EC", line=dict(color="purple")
                    ))
                fig2.update_layout(
                    title="pH·EC 일별 평균", xaxis_title="날짜", yaxis_title="값",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
                )
                st.plotly_chart(fig2, width="stretch")
        else:
            st.info("환경 데이터가 없어요.")

        with st.expander("📂 환경 데이터 원본 및 다운로드"):
            if env_df is not None and not env_df.empty:
                st.dataframe(env_df, width="stretch")
                csv = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("환경 데이터 CSV 다운로드", csv, file_name="env_data.csv", mime="text/csv")
            else:
                st.info("환경 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ============================================================
# 탭 3: 생육 결과
# ============================================================
with tab3:
    try:
        st.subheader("📊 생육 결과 분석")

        st.subheader("🏆 핵심 결과")
        col1, col2, col3 = st.columns(3)
        with col1:
            if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
                avg_gw = growth_df["생중량"].mean()
                max_gw = growth_df["생중량"].max()
                min_gw = growth_df["생중량"].min()
                max_idx = growth_df["생중량"].idxmax()
                min_idx = growth_df["생중량"].idxmin()
                st.metric("평균 생중량 (g)", f"{avg_gw:.3f}")
                st.metric("최대 생중량 (g)", f"{max_gw:.3f}", f"개체 {growth_df.loc[max_idx, '개체번호']}")
                st.metric("최소 생중량 (g)", f"{min_gw:.3f}", f"개체 {growth_df.loc[min_idx, '개체번호']}")
            else:
                st.metric("평균 생중량 (g)", "데이터 없음")

        st.subheader("📈 개체별 생육 비교 (막대 그래프)")
        if growth_df is not None and not growth_df.empty:
            if "생중량" in growth_df.columns:
                gw_mean = growth_df.groupby("개체번호")["생중량"].mean().reset_index().sort_values("생중량", ascending=False)
                render_bar_chart(gw_mean, "개체번호", "평균 생중량 (g)", "개체별 평균 생중량 (⭐ 가장 중요)")

            if "잎 수" in growth_df.columns and "잎 수" in selected_metrics:
                leaf_mean = growth_df.groupby("개체번호")["잎 수"].mean().reset_index()
                render_bar_chart(leaf_mean, "개체번호", "평균 잎 수 (장)", "개체별 평균 잎 수")

            if "지상부 길이" in growth_df.columns and "지상부 길이" in selected_metrics:
                shoot_mean = growth_df.groupby("개체번호")["지상부 길이"].mean().reset_index()
                render_bar_chart(shoot_mean, "개체번호", "평균 지상부 길이 (mm)", "개체별 평균 지상부 길이")

            if "지하부 길이" in growth_df.columns and "지하부 길이" in selected_metrics:
                root_mean = growth_df.groupby("개체번호")["지하부 길이"].mean().reset_index()
                render_bar_chart(root_mean, "개체번호", "평균 지하부 길이 (mm)", "개체별 평균 지하부 길이")

        st.subheader("📦 생중량 분포")
        if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
            gw_vals = growth_df["생중량"].dropna()
            if len(gw_vals) >= 1:
                render_boxplot(growth_df, "생중량", "개체별 생체중량 분포")
            else:
                st.info("생중량 데이터가 없어요.")
        else:
            st.info("생중량 데이터가 없어요.")

        st.subheader("🔗 잎 수·지상부 길이와 생중량의 관계")
        if growth_df is not None and not growth_df.empty:
            if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                render_scatter_trendline(
                    growth_df["잎 수"], growth_df["생중량"],
                    "잎 수 (장)", "생중량 (g)", "잎 수 vs 생중량"
                )
            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                render_scatter_trendline(
                    growth_df["지상부 길이"], growth_df["생중량"],
                    "지상부 길이 (mm)", "생중량 (g)", "지상부 길이 vs 생중량"
                )

        st.subheader("⚖️ 생체중량 상위 25% vs 하위 25% 비교")
        if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
            gw_clean = growth_df["생중량"].dropna()
            if len(gw_clean) >= 4:
                q75 = gw_clean.quantile(0.75)
                q25 = gw_clean.quantile(0.25)
                top_mask = growth_df["생중량"].notna() & (growth_df["생중량"] >= q75)
                bottom_mask = growth_df["생중량"].notna() & (growth_df["생중량"] <= q25)
                top_df = growth_df[top_mask]
                bottom_df = growth_df[bottom_mask]

                comp_data = {
                    "구분": ["상위 25%", "하위 25%"],
                    "개체 수": [len(top_df), len(bottom_df)],
                    "평균 생중량 (g)": [
                        round(top_df["생중량"].mean(), 3) if not top_df.empty else 0,
                        round(bottom_df["생중량"].mean(), 3) if not bottom_df.empty else 0
                    ],
                    "평균 잎 수 (장)": [
                        round(top_df["잎 수"].mean(), 2) if (not top_df.empty and "잎 수" in top_df.columns) else "N/A",
                        round(bottom_df["잎 수"].mean(), 2) if (not bottom_df.empty and "잎 수" in bottom_df.columns) else "N/A"
                    ],
                    "평균 지상부 길이 (mm)": [
                        round(top_df["지상부 길이"].mean(), 2) if (not top_df.empty and "지상부 길이" in top_df.columns) else "N/A",
                        round(bottom_df["지상부 길이"].mean(), 2) if (not bottom_df.empty and "지상부 길이" in bottom_df.columns) else "N/A"
                    ],
                }
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, width="stretch")
            else:
                st.info("상위 25% vs 하위 25% 비교를 위해 데이터가 4개 이상 필요해요.")

        with st.expander("📂 생육 데이터 원본 및 다운로드"):
            if growth_df is not None and not growth_df.empty:
                st.dataframe(growth_df, width="stretch")
                csv = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("생육 데이터 CSV 다운로드", csv, file_name="growth_data.csv", mime="text/csv")
            else:
                st.info("생육 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
