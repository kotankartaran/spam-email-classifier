import streamlit as st
import joblib

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("📋 Project Information")

st.sidebar.success("""
Spam Email Classifier

🤖 Algorithm:
Multinomial Naive Bayes

📊 Dataset:
SMS Spam Collection Dataset

📈 Accuracy:
97.4%

🛠 Technologies:
• Python
• Scikit-Learn
• NLP
• TF-IDF
• Streamlit
""")

# ==========================
# MAIN PAGE
# ==========================

st.title("📧 Spam Email Detection System")

st.markdown(
    "Detect whether a message is Spam or Legitimate using Machine Learning."
)

st.divider()

# ==========================
# INPUT
# ==========================

message = st.text_area(
    "Enter Email / SMS Message",
    height=200,
    placeholder="Congratulations! You have won $1000..."
)

# ==========================
# PREDICT
# ==========================

if st.button(
    "🔍 Analyze Message",
    width="stretch"
):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        transformed = vectorizer.transform([message])

        prediction = model.predict(transformed)[0]

        probability = model.predict_proba(transformed)[0]

        ham_prob = probability[0]
        spam_prob = probability[1]

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam Probability",
                f"{spam_prob*100:.2f}%"
            )

        with col2:
            st.metric(
                "Safe Probability",
                f"{ham_prob*100:.2f}%"
            )

        st.progress(float(spam_prob))

        if prediction == 1:
            st.error(
                f"🚨 Spam Detected ({spam_prob*100:.2f}% confidence)"
            )
        else:
            st.success(
                f"✅ Legitimate Message ({ham_prob*100:.2f}% confidence)"
            )

        st.subheader("Message Preview")

        st.info(message)