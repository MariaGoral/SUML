import streamlit as st

from model.predict import predict_job_offer
from model.model_utils import ModelLoadError

st.set_page_config(
    page_title="Fake Job Offer Detector",
    page_icon="🕵️",
    layout="wide"
)

# ======================
# CSS
# ======================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom, #0f172a, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.info-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}

.stTextArea textarea {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 10px !important;
}

.stButton button {
    background-color: #06b6d4;
    color: white;
    border-radius: 10px;
    border: none;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #0891b2;
}

.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🕵️ Fake Job Offer Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered detection of fraudulent job postings</div>',
    unsafe_allow_html=True
)

st.subheader("📄 Treść ogłoszenia")

job_text = st.text_area(
    "",
    height=300,
    placeholder="""
Senior Python Developer

We are looking for talented developers...
"""
)


if st.button("🔍 Analyze Job Offer"):

    if not job_text.strip():

        st.warning("Wklej treść ogłoszenia.")

    else:

        try:

            with st.spinner("Analizowanie treści..."):

                result = predict_job_offer(job_text)

            st.success("Analiza zakończona.")

            prediction = result["prediction"]
            confidence = result["confidence"] * 100

            if prediction == 1:

                st.error("⚠️ Oferta może być fałszywa")

                fake_prob = confidence
                real_prob = 100 - confidence

            else:

                st.success("✅ Oferta wygląda na prawdziwą")

                real_prob = confidence
                fake_prob = 100 - confidence

            st.progress(int(fake_prob))

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fake Probability",
                    f"{fake_prob:.2f}%"
                )

            with col2:
                st.metric(
                    "Real Probability",
                    f"{real_prob:.2f}%"
                )

            st.write("### Szczegóły predykcji")

            st.json(result)

        except ModelLoadError as e:

            st.error(
                f"Nie można załadować modelu: {e}"
            )

        except ValueError as e:

            st.warning(str(e))

        except Exception as e:

            st.error(
                f"Wystąpił nieoczekiwany błąd: {e}"
            )


st.divider()

st.caption(
    "Fake Job Offer Detector | SUML Project 2025/2026"
)