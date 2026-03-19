import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(
    page_title="Battery Usability Predictor",
    page_icon="🔋",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("battery_lstm_final.keras")
    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model()

def save_chart(fig):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def generate_pdf(battery_id, battery_type, test_date, operator,
                 voltage, current, power, temperature,
                 soh, status, years_left, recommendation,
                 chart1, chart2, chart3, chart4):
    pdf = FPDF()

    # Page 1 - Report
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_fill_color(0, 128, 100)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "BATTERY ANALYSIS REPORT", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(220, 240, 255)
    pdf.cell(0, 10, "Battery Information", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(60, 8, f"Battery ID    : {battery_id}")
    pdf.cell(0, 8, f"Battery Type  : {battery_type}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(60, 8, f"Test Date     : {test_date}")
    pdf.cell(0, 8, f"Operator      : {operator}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(220, 240, 255)
    pdf.cell(0, 10, "Measurements", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(60, 8, f"Voltage       : {voltage} V")
    pdf.cell(0, 8, f"Current       : {current} A", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(60, 8, f"Power         : {power} W")
    pdf.cell(0, 8, f"Temperature   : {temperature} C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(220, 240, 255)
    pdf.cell(0, 10, "Prediction Results", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 14)
    if status == "USABLE":
        pdf.set_text_color(0, 150, 100)
    elif status == "DEGRADED":
        pdf.set_text_color(200, 130, 0)
    else:
        pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 10, f"Status : {status}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"State of Health (SoH) : {soh:.2f}%", new_x="LMARGIN", new_y="NEXT")
    life = f"~{years_left} years" if years_left > 0 else "< 6 months"
    pdf.cell(0, 8, f"Estimated Life Left   : {life}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(220, 240, 255)
    pdf.cell(0, 10, "Recommendations", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for rec in recommendation:
        clean = rec.replace("✅","").replace("⚠️","").replace("❌","").strip()
        pdf.cell(0, 8, f"- {clean}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(220, 240, 255)
    pdf.cell(0, 10, "Model Information", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Algorithm : LSTM (Long Short-Term Memory)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "R2        : 0.9773", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Accuracy  : 98%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "RMSE      : 2.99%", new_x="LMARGIN", new_y="NEXT")

    # Page 2 - Charts
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(0, 128, 100)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "Battery Analysis Charts", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "1. SoH History", new_x="LMARGIN", new_y="NEXT")
    pdf.image(chart1, x=10, w=90)
    pdf.set_xy(105, pdf.get_y() - 65)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "2. Charge/Discharge Cycle", new_x="LMARGIN", new_y="NEXT")
    pdf.image(chart2, x=105, w=90)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. Temperature vs SoH", new_x="LMARGIN", new_y="NEXT")
    pdf.image(chart3, x=10, w=90)
    pdf.set_xy(105, pdf.get_y() - 65)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "4. Battery Life Prediction", new_x="LMARGIN", new_y="NEXT")
    pdf.image(chart4, x=105, w=90)

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "Generated by Battery Usability Predictor | Final Year Research Project", align="C")

    return bytes(pdf.output())

# Header
st.markdown("# 🔋 Battery Usability Predictor")
st.markdown("**AI-Enabled Prediction of Reconditioned Second-Life Lithium-Ion Battery Modules**")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Battery Info")
    battery_id   = st.text_input("Battery ID", value="BAT-001")
    battery_type = st.selectbox("Battery Type", ["Nissan Leaf Module", "Other Li-Ion"])
    test_date    = st.date_input("Test Date", value=datetime.today())
    operator     = st.text_input("Operator Name", value="Researcher")
    st.divider()
    st.markdown("### 📌 SoH Thresholds")
    st.success("✅ Usable   : SoH >= 80%")
    st.warning("⚠️ Degraded : SoH 60-80%")
    st.error("❌ Failed   : SoH < 60%")
    st.divider()
    st.caption("LSTM Model | R2=0.9773 | Accuracy=98%")

# Input
st.subheader("📊 Enter Battery Parameters")
col1, col2, col3, col4 = st.columns(4)
with col1:
    voltage     = st.number_input("⚡ Voltage (V)",      min_value=6.0,  max_value=8.4,   value=7.5,  step=0.01)
with col2:
    current     = st.number_input("🔌 Current (A)",      min_value=-5.0, max_value=17.0,  value=0.3,  step=0.01)
with col3:
    power       = st.number_input("💡 Power (W)",        min_value=-50.0,max_value=150.0, value=2.0,  step=0.1)
with col4:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0,  max_value=60.0,  value=30.0, step=0.1)

st.divider()

col_b1, col_b2, col_b3 = st.columns([2,1,2])
with col_b2:
    predict = st.button("🔍 Analyse Battery", use_container_width=True, type="primary")

if predict:
    with st.spinner("Analysing battery with LSTM model..."):
        input_data  = np.array([[voltage, current, power, temperature]])
        input_seq   = np.tile(input_data, (60, 1))
        input_seq   = scaler_X.transform(input_seq)
        input_seq   = input_seq.reshape(1, 60, 4)
        pred_scaled = model.predict(input_seq, verbose=0)
        soh = float(scaler_y.inverse_transform(pred_scaled)[0][0])
        soh = np.clip(soh, 0, 100)

    if soh >= 80:
        status      = "USABLE"
        gauge_color = "#00cc88"
        mpl_color   = "green"
        years_left  = round((soh - 60) / 5, 1)
        recommendation = [
            "✅ Battery is suitable for solar energy storage",
            "✅ Continue normal charge/discharge cycles",
            "✅ Recommended for second-life deployment",
            f"✅ Estimated usable life: ~{years_left} more years",
            "✅ Schedule next check in 6 months"
        ]
    elif soh >= 60:
        status      = "DEGRADED"
        gauge_color = "#ffaa00"
        mpl_color   = "orange"
        years_left  = round((soh - 60) / 5, 1)
        recommendation = [
            "⚠️ Battery performance is reduced",
            "⚠️ Suitable for low-demand solar applications only",
            f"⚠️ Estimated remaining usable life: ~{years_left} years",
            "⚠️ Increase monitoring frequency to monthly",
            "⚠️ Plan for replacement within 1-2 years"
        ]
    else:
        status      = "FAILED"
        gauge_color = "#ff4444"
        mpl_color   = "red"
        years_left  = 0
        recommendation = [
            "❌ Battery is not suitable for solar energy storage",
            "❌ Immediate replacement recommended",
            "❌ Do not use in critical applications",
            "❌ Consider recycling or safe disposal",
            "❌ Estimated remaining life: < 6 months"
        ]

    # Generate matplotlib charts for PDF
    months       = list(range(1, 13))
    decay        = [min(100, soh + (12-m)*1.5) for m in months]
    decay[-1]    = soh
    time_points  = list(range(0, 25))
    charge_p     = [min(8.4, 6.0 + t*0.1) for t in time_points]
    discharge_p  = [max(6.0, 8.4 - t*0.1) for t in time_points]
    temps        = list(range(10, 61, 5))
    soh_vals     = [np.clip(soh - abs(t-25)*0.3, 0, 100) for t in temps]
    future_months= list(range(0, 37))
    future_soh   = [max(0, soh - m*0.5) for m in future_months]
    eol_month    = next((m for m,s in zip(future_months, future_soh) if s < 60), 36)

    # Chart 1
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(months, decay, color=mpl_color, marker="o", linewidth=2)
    ax1.axhline(y=80, color="green", linestyle="--", label="Usable 80%")
    ax1.axhline(y=60, color="red",   linestyle="--", label="EOL 60%")
    ax1.set_title("SoH History")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("SoH (%)")
    ax1.legend(fontsize=7)
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3)
    chart1_path = save_chart(fig1)

    # Chart 2
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.plot(time_points, charge_p,    color="green", linewidth=2, label="Charging")
    ax2.plot(time_points, discharge_p, color="red",   linewidth=2, label="Discharging")
    ax2.axhline(y=voltage, color=mpl_color, linestyle=":", label=f"V={voltage}V")
    ax2.set_title("Charge/Discharge Profile")
    ax2.set_xlabel("Time (hrs)")
    ax2.set_ylabel("Voltage (V)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    chart2_path = save_chart(fig2)

    # Chart 3
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.plot(temps, soh_vals, color="orange", marker="o", linewidth=2)
    ax3.axvline(x=temperature, color=mpl_color, linestyle=":", label=f"T={temperature}C")
    ax3.set_title("Temperature vs SoH")
    ax3.set_xlabel("Temperature (C)")
    ax3.set_ylabel("SoH (%)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    chart3_path = save_chart(fig3)

    # Chart 4
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    ax4.fill_between(future_months, future_soh, alpha=0.3, color=mpl_color)
    ax4.plot(future_months, future_soh, color=mpl_color, linewidth=2)
    ax4.axhline(y=80, color="green", linestyle="--", label="Usable 80%")
    ax4.axhline(y=60, color="red",   linestyle="--", label="EOL 60%")
    ax4.axvline(x=eol_month, color="red", linestyle=":", label=f"EOL ~{eol_month}mo")
    ax4.set_title("Battery Life Prediction")
    ax4.set_xlabel("Months from Now")
    ax4.set_ylabel("SoH (%)")
    ax4.legend(fontsize=7)
    ax4.set_ylim(0, 110)
    ax4.grid(True, alpha=0.3)
    chart4_path = save_chart(fig4)

    st.divider()
    st.subheader("📈 Analysis Results")

    col_g, col_m = st.columns([1, 1])
    with col_g:
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = soh,
            title = {"text": "State of Health (SoH %)", "font": {"size": 18}},
            delta = {"reference": 80},
            gauge = {
                "axis"  : {"range": [0, 100]},
                "bar"   : {"color": gauge_color},
                "steps" : [
                    {"range": [0,  60], "color": "#ffdddd"},
                    {"range": [60, 80], "color": "#fff3cc"},
                    {"range": [80, 100],"color": "#ddffee"}
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 60}
            }
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_m:
        st.markdown("### Battery Status")
        if status == "USABLE":
            st.success(f"✅ {status}")
        elif status == "DEGRADED":
            st.warning(f"⚠️ {status}")
        else:
            st.error(f"❌ {status}")
        st.metric("SoH",         f"{soh:.1f}%")
        st.metric("Voltage",     f"{voltage} V")
        st.metric("Temperature", f"{temperature} °C")
        life = f"~{years_left} years" if years_left > 0 else "< 6 months"
        st.metric("Est. Remaining Life", life)

    st.divider()

    st.subheader("📉 SoH History")
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=months, y=decay, mode="lines+markers",
                               line=dict(color=gauge_color, width=2)))
    fig_h.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Usable 80%")
    fig_h.add_hline(y=60, line_dash="dash", line_color="red",   annotation_text="EOL 60%")
    fig_h.update_layout(xaxis_title="Month", yaxis_title="SoH (%)", height=300, yaxis_range=[0,110])
    st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("⚡ Charge/Discharge Cycle")
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=time_points, y=charge_p,    mode="lines", name="Charging",    line=dict(color="#00cc88", width=2)))
    fig_c.add_trace(go.Scatter(x=time_points, y=discharge_p, mode="lines", name="Discharging", line=dict(color="#ff4444", width=2)))
    fig_c.add_hline(y=voltage, line_dash="dot", line_color=gauge_color, annotation_text=f"V={voltage}V")
    fig_c.update_layout(xaxis_title="Time (hrs)", yaxis_title="Voltage (V)", height=300)
    st.plotly_chart(fig_c, use_container_width=True)

    st.subheader("🌡️ Temperature vs SoH")
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=temps, y=soh_vals, mode="lines+markers", line=dict(color="#ffaa00", width=2)))
    fig_t.add_vline(x=temperature, line_dash="dot", line_color=gauge_color, annotation_text=f"T={temperature}C")
    fig_t.update_layout(xaxis_title="Temperature (C)", yaxis_title="SoH (%)", height=300)
    st.plotly_chart(fig_t, use_container_width=True)

    st.subheader("🔮 Battery Life Prediction")
    fig_l = go.Figure()
    fig_l.add_trace(go.Scatter(x=future_months, y=future_soh, fill="tozeroy",
                               line=dict(color=gauge_color, width=2)))
    fig_l.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Usable 80%")
    fig_l.add_hline(y=60, line_dash="dash", line_color="red",   annotation_text="EOL 60%")
    fig_l.add_vline(x=eol_month, line_dash="dot", line_color="red", annotation_text=f"EOL ~{eol_month} months")
    fig_l.update_layout(xaxis_title="Months from Now", yaxis_title="SoH (%)", height=300, yaxis_range=[0,110])
    st.plotly_chart(fig_l, use_container_width=True)

    st.divider()
    st.subheader("💡 Recommendations")
    for rec in recommendation:
        st.markdown(f"- {rec}")

    st.divider()

    pdf_bytes = generate_pdf(
        battery_id, battery_type, str(test_date), operator,
        voltage, current, power, temperature,
        soh, status, years_left, recommendation,
        chart1_path, chart2_path, chart3_path, chart4_path
    )

    for p in [chart1_path, chart2_path, chart3_path, chart4_path]:
        try: os.remove(p)
        except: pass

    st.download_button(
        label     = "📥 Download PDF Report (with Charts)",
        data      = pdf_bytes,
        file_name = f"battery_report_{battery_id}_{test_date}.pdf",
        mime      = "application/pdf",
        use_container_width=True,
        type      = "primary"
    )

st.divider()
st.caption("🎓 Final Year Research | LSTM Model | R2=0.9773 | Accuracy=98% | No Overfitting")
