import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef5ff, #ffffff, #f3efff);
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #173b73 !important;
    margin-top: 10px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    font-weight: 500;
    color: #405777 !important;
    margin-bottom: 25px;
}

.accuracy-card {
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    background: linear-gradient(135deg, #ffffff, #eaf3ff);
    box-shadow: 0 5px 20px rgba(0,0,0,0.10);
    border: 1px solid #cbdcf5;
    margin-bottom: 25px;
    color: #173b73 !important;
}

.accuracy-number {
    font-size: 34px;
    font-weight: 800;
    color: #2563eb !important;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #173b73 !important;
    margin-top: 15px;
    margin-bottom: 15px;
}

label {
    color: #173b73 !important;
    font-weight: 600 !important;
}

[data-baseweb="input"] {
    background-color: #ffffff !important;
    border-radius: 10px !important;
}

[data-baseweb="input"] input {
    color: #173b73 !important;
    background-color: #ffffff !important;
}

[data-baseweb="select"] {
    background-color: #ffffff !important;
}

[data-baseweb="select"] * {
    color: #173b73 !important;
}

div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    font-size: 19px;
    font-weight: 700;
    border: none;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white !important;
}

.result-approved {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    background: linear-gradient(135deg, #e7fff1, #d8ffea);
    border: 2px solid #65d89a;
    color: #176b3a !important;
    font-size: 25px;
    font-weight: 700;
}

.result-rejected {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    background: linear-gradient(135deg, #fff0f0, #ffe2e2);
    border: 2px solid #ff8c8c;
    color: #a32121 !important;
    font-size: 25px;
    font-weight: 700;
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 15px;
    font-size: 14px;
    color: #405777 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/loan_approval_model.pkl")

# ---------------- LOAD DATASET ----------------
data = pd.read_csv("dataset/loan_approval_dataset.csv")
data.columns = data.columns.str.strip()

# ---------------- ENCODING ----------------
encoder = LabelEncoder()

for column in data.select_dtypes(include="object").columns:
    data[column] = encoder.fit_transform(data[column].astype(str))

# ---------------- DATA SPLIT ----------------
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- ACCURACY ----------------
accuracy = model.score(X_test, y_test)


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🏦 Loan Approval AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">✨ Smart Loan Approval Prediction using Machine Learning ✨</div>',
    unsafe_allow_html=True
)


# ---------------- ACCURACY CARD ----------------
st.markdown(
    f"""
    <div class="accuracy-card">
        <div>🎯 MODEL TEST ACCURACY</div>
        <div class="accuracy-number">{accuracy * 100:.2f}%</div>
        <div>🌳 Random Forest Classifier</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- APPLICANT DETAILS ----------------
st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    loan_id = st.number_input(
        "🆔 Loan ID",
        min_value=1,
        value=1
    )

    no_of_dependents = st.number_input(
        "👨‍👩‍👧 Number of Dependents",
        min_value=0,
        max_value=10,
        value=2
    )

    education = st.selectbox(
        "🎓 Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "💼 Self Employed",
        ["No", "Yes"]
    )

    income_annum = st.number_input(
        "💰 Annual Income",
        min_value=0,
        value=500000
    )

    loan_amount = st.number_input(
        "💵 Loan Amount",
        min_value=0,
        value=500000
    )


with col2:

    loan_term = st.number_input(
        "📅 Loan Term (months)",
        min_value=1,
        value=12
    )

    cibil_score = st.number_input(
        "📊 CIBIL Score",
        min_value=0,
        max_value=900,
        value=700
    )

    residential_assets_value = st.number_input(
        "🏠 Residential Assets Value",
        min_value=0,
        value=500000
    )

    commercial_assets_value = st.number_input(
        "🏢 Commercial Assets Value",
        min_value=0,
        value=0
    )

    luxury_assets_value = st.number_input(
        "✨ Luxury Assets Value",
        min_value=0,
        value=500000
    )

    bank_asset_value = st.number_input(
        "🏦 Bank Asset Value",
        min_value=0,
        value=500000
    )


# ---------------- PREDICT BUTTON ----------------

st.markdown("---")

predict = st.button("🔮 Predict Loan Approval")


if predict:

    # Convert text values into numbers
    education_value = 0 if education == "Graduate" else 1
    self_employed_value = 0 if self_employed == "No" else 1

    # Create input dataframe
    input_data = pd.DataFrame([[
        loan_id,
        no_of_dependents,
        education_value,
        self_employed_value,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=[
        "loan_id",
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ])

    # Prediction
    prediction = model.predict(input_data)[0]

    # ---------------- RESULT ----------------

    if prediction == 0:

        st.balloons()

        st.markdown(
            """
            <div class="result-approved">
                🎉 Loan Approved! 🎉<br>
                <span style="font-size:18px;">
                The applicant is predicted to be eligible for the loan.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-rejected">
                ❌ Loan Rejected<br>
                <span style="font-size:18px;">
                The applicant is predicted to be not eligible for the loan.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------- FOOTER ----------------

st.markdown(
    """
    <div class="footer">
        🤖 Machine Learning Project • Random Forest • Streamlit<br>
        Made for Academic Project Presentation
    </div>
    """,
    unsafe_allow_html=True
)