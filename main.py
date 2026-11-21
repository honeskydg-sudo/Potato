import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# ============================
# 한글 폰트 및 CSS 설정
# ============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ============================
# CSV 읽기 + 열 이름 표준화
# ============================
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

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

@st.cache_data
def load_data(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            return std_cols(df)
        except Exception:
            continue
    return pd.DataFrame()

# ============================
# 산점도 + 추세선 (공통 함수)
# ============================
def make_scatter_trendline(df, x_col, y_col, title=None):
    valid = df[[x_col, y_col]].dropna()
    if len(valid) < 2:
        return None
    x = valid[x_col].values
    y = valid[y_col].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="데이터", marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=x, y=p(x), mode="lines", name="추세선", line=dict(dash="dash", color="red")))

    fig.update_layout(
        title=title or f"{x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        hovermode="closest"
    )
    return fig

# ============================
# 파일 업로드
# ============================
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 CSV", type="csv")

school_name = growth_file.name.split(".")[0] if growth_file is not None else "미지정"

st.title("극지식물 개체별 생육 차이 분석 대시보드")
if growth_file is None:
    st.info("생육 CSV 파일을 업로드하면 개체별 생육 차이 분석 대시보드를 볼 수 있어요. (환경 데이터는 현재 사용하지 않아요)")
else:
    st.caption(f"학교명: {school_name}")

# ============================
# 데이터 로딩 (한 번만 읽기)
# ============================
if env_file is not None:
    with st.spinner("환경 데이터를 읽고 있어요..."):
        env_df = load_data(env_file)
    st.sidebar.info("환경 데이터는 현재 분석에서 사용하지 않아요. (CSV만 업로드 가능)")

if growth_file is not None:
    with st.spinner("생육 데이터를 읽고 있어요..."):
        growth_df = load_data(growth_file)
else:
    growth_df = None

# ============================
# 사이드바: 생육 측정 항목 선택
# ============================
st.sidebar.header("생육 측정 항목 선택")
if growth_df is not None and len(growth_df) > 0:
    num_cols = [c for c in growth_df.select_dtypes(include=[np.number]).columns if c != "개체번호"]
    weight_cols = [c for c in ["생중량", "지상부 질량", "지하부 질량"] if c in num_cols]
    size_cols = [c for c in ["지상부 길이", "지하부 길이"] if c in num_cols]
    leaf_cols = ["잎 수"] if "잎 수" in num_cols else []

    weight_col = st.sidebar.selectbox("체중 지표", ["전체"] + weight_cols, index=0)
    size_col = st.sidebar.selectbox("크기 지표", ["전체"] + size_cols, index=0)
    leaf_col = st.sidebar.selectbox("잎 수", ["전체"] + leaf_cols, index=0)
else:
    weight_col = "전체"
    size_col = "전체"
    leaf_col = "전체"

# ============================
# 탭 구성
# ============================
tab1, tab2, tab3 = st.tabs(["개체별 생육 요약", "특성 간 관계 분석", "상위·하위 개체 비교"])

# ============================
# 탭 1: 개체별 생육 요약
# ============================
with tab1:
    try:
        if growth_df is None or len(growth_df) < 2:
            st.info("생육 데이터를 업로드하면 개체별 생육 요약을 볼 수 있어요. (데이터 2개 이상 필요)")
        else:
            num_cols = [c for c in growth_df.select_dtypes(include=[np.number]).columns if c != "개체번호"]
            if len(num_cols) == 0:
                st.info("분석에 사용할 수 있는 숫자 열이 없어요. CSV 열 이름을 확인해 주세요.")
            else:
                st.subheader("📊 개체별 생육 요약 통계")

                stats = {}
                if "잎 수" in num_cols:
                    stats["잎 수 (장)"] = f"평균 {growth_df['잎 수'].mean():.1f}장 / 표준편차 {growth_df['잎 수'].std():.1f}장"
                if "지상부 길이" in num_cols:
                    stats["지상부 길이 (mm)"] = f"평균 {growth_df['지상부 길이'].mean():.1f}mm / 표준편차 {growth_df['지상부 길이'].std():.1f}mm"
                if "지하부 길이" in num_cols:
                    stats["지하부 길이 (mm)"] = f"평균 {growth_df['지하부 길이'].mean():.1f}mm / 표준편차 {growth_df['지하부 길이'].std():.1f}mm"
                if "지상부 질량" in num_cols:
                    stats["지상부 질량 (g)"] = f"평균 {growth_df['지상부 질량'].mean():.2f}g / 표준편차 {growth_df['지상부 질량'].std():.2f}g"
                if "지하부 질량" in num_cols:
                    stats["지하부 질량 (g)"] = f"평균 {growth_df['지하부 질량'].mean():.2f}g / 표준편차 {growth_df['지하부 질량'].std():.2f}g"
                if "생중량" in num_cols:
                    stats["생중량 (g)"] = f"평균 {growth_df['생중량'].mean():.2f}g / 표준편차 {growth_df['생중량'].std():.2f}g"

                cols = st.columns(len(stats))
                for i, (k, v) in enumerate(stats.items()):
                    cols[i].metric(k, v)

                st.subheader("📋 전체 개체 생육 표")
                display_cols = ["개체번호"] + num_cols
                display_df = growth_df[display_cols].copy()
                display_df["개체번호"] = display_df["개체번호"].astype(str)
                st.dataframe(display_df, width="stretch")

                csv = display_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 표 다운로드 (CSV)", data=csv, file_name=f"{school_name}_개체별_생육.csv", mime="text/csv")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ============================
# 탭 2: 특성 간 관계 분석
# ============================
with tab2:
    try:
        if growth_df is None or len(growth_df) < 2:
            st.info("생육 데이터를 업로드하면 특성 간 관계를 볼 수 있어요. (데이터 2개 이상 필요)")
        else:
            num_cols = [c for c in growth_df.select_dtypes(include=[np.number]).columns if c != "개체번호"]
            if len(num_cols) < 2:
                st.info("비교할 수 있는 특성 열이 2개 이상 필요해요.")
            else:
                st.subheader("📈 특성 간 관계 분석")

                figs = []
                for i in range(len(num_cols)):
                    for j in range(i+1, len(num_cols)):
                        x_col = num_cols[i]
                        y_col = num_cols[j]
                        fig = make_scatter_trendline(growth_df, x_col, y_col)
                        if fig is not None:
                            figs.append(fig)

                if len(figs) == 0:
                    st.info("추세선을 그릴 수 있는 데이터가 부족해요. (각 조합당 2개 이상 필요)")
                else:
                    n = len(figs)
                    rows = (n + 2) // 3
                    cols = 3 if n > 2 else 1
                    fig_grid = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{s.layout.title.text}" for s in figs])
                    for idx, s in enumerate(figs):
                        r = idx // 3 + 1
                        c = idx % 3 + 1
                        for trace in s.data:
                            fig_grid.add_trace(trace, row=r, col=c)
                    fig_grid.update_layout(height=400 * rows, font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
                    st.plotly_chart(fig_grid, width="stretch")

                st.subheader("🔗 특성 간 상관관계")
                corr = growth_df[num_cols].corr()
                st.write(corr.round(2))

                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale="RdBu",
                    text=np.round(corr.values, 2),
                    texttemplate="%{text}"
                ))
                fig_corr.update_layout(
                    title="상관관계 히트맵",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
                )
                st.plotly_chart(fig_corr, width="stretch")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ============================
# 탭 3: 상위·하위 개체 비교
# ============================
with tab3:
    try:
        if growth_df is None or len(growth_df) < 2:
            st.info("생육 데이터를 업로드하면 상위·하위 개체를 비교할 수 있어요. (데이터 2개 이상 필요)")
        else:
            num_cols = [c for c in growth_df.select_dtypes(include=[np.number]).columns if c != "개체번호"]

            if "생중량" not in num_cols and {"지상부 질량", "지하부 질량"} <= set(num_cols):
                growth_df = growth_df.copy()
                growth_df["생중량"] = growth_df["지상부 질량"] + growth_df["지하부 질량"]
                num_cols = [c for c in growth_df.select_dtypes(include=[np.number]).columns if c != "개체번호"]

            if "생중량" not in num_cols:
                st.info("체중 지표를 계산할 수 있는 열이 없어요.")
            else:
                sorted_df = growth_df.sort_values("생중량", ascending=False)
                top5 = sorted_df.head(5)
                bottom5 = sorted_df.tail(5)

                st.subheader("🏆 생중량 상위 5개 개체")
                top_cols = ["개체번호"] + [c for c in ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"] if c in top5.columns]
                top_display = top5[top_cols].copy()
                top_display["개체번호"] = top_display["개체번호"].astype(str)
                st.dataframe(top_display, width="stretch")

                st.subheader("🥉 생중량 하위 5개 개체")
                bottom_cols = ["개체번호"] + [c for c in ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"] if c in bottom5.columns]
                bottom_display = bottom5[bottom_cols].copy()
                bottom_display["개체번호"] = bottom_display["개체번호"].astype(str)
                st.dataframe(bottom_display, width="stretch")

                st.subheader("📊 상위 5 vs 하위 5 개체 생중량 비교")

                compare_df = pd.concat([
                    top5[["개체번호", "생중량"]].assign(구분="상위 5"),
                    bottom5[["개체번호", "생중량"]].assign(구분="하위 5")
                ])
                compare_df["개체번호"] = compare_df["개체번호"].astype(str)

                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=compare_df[compare_df["구분"] == "상위 5"]["생중량"], name="상위 5개", marker_color="green", boxmean=True))
                fig_box.add_trace(go.Box(y=compare_df[compare_df["구분"] == "하위 5"]["생중량"], name="하위 5개", marker_color="orange", boxmean=True))
                fig_box.update_layout(
                    title="상위 5 vs 하위 5 개체 생중량 비교",
                    yaxis_title="생중량(g)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
                )
                st.plotly_chart(fig_box, width="stretch")

                fig_individual = go.Figure()
                all_names = list(top5["개체번호"].astype(str)) + list(bottom5["개체번호"].astype(str))
                all_vals = list(top5["생중량"]) + list(bottom5["생중량"])
                all_colors = ["green"] * len(top5) + ["orange"] * len(bottom5)
                fig_individual.add_trace(go.Bar(x=all_names, y=all_vals, marker_color=all_colors))
                fig_individual.update_layout(
                    title="상위 5 vs 하위 5 개체 개별 생중량",
                    xaxis_title="개체",
                    yaxis_title="생중량(g)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_individual, width="stretch")

                st.subheader("📋 상위·하위 지표 요약")
                compare_rows = []
                for col in ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"]:
                    if col in num_cols:
                        top_mean = top5[col].mean()
                        bottom_mean = bottom5[col].mean()
                        compare_rows.append({"지표": col, "상위 5 평균": top_mean, "하위 5 평균": bottom_mean})

                if compare_rows:
                    comp_df = pd.DataFrame(compare_rows)
                    st.dataframe(comp_df, width="stretch")

                    norm = comp_df.copy()
                    for c in ["상위 5 평균", "하위 5 평균"]:
                        norm[c] = norm[c] / norm[c].max()

                    fig_norm = go.Figure(data=[
                        go.Bar(name="상위 5 평균", x=norm["지표"], y=norm["상위 5 평균"], marker_color="green"),
                        go.Bar(name="하위 5 평균", x=norm["지표"], y=norm["하위 5 평균"], marker_color="orange")
                    ])
                    fig_norm.update_layout(
                        barmode="group",
                        title="상위 5 vs 하위 5 지표 정규화 비교",
                        xaxis_title="지표",
                        yaxis_title="정규화 값 (상위 최대=1)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
                    )
                    st.plotly_chart(fig_norm, width="stretch")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ============================
# 바닥글
# ============================
st.divider()
st.caption("본 대시보드는 극지식물 개체별 생육 데이터를 시각화한 교육용 자료예요.")
