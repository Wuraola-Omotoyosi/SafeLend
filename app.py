import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("safelend_logreg.pkl")

st.title("💳 SafeLend - Loan Default Risk Predictor")
st.write("Predict whether a borrower is likely to default based on input details.")

# Collect user inputs
age = st.number_input("Borrower Age", min_value=18, max_value=80, step=1)
loanamount = st.number_input("Loan Amount", min_value=1000, max_value=1000000, step=500)
termdays = st.number_input("Loan Term (days)", min_value=7, max_value=365, step=1)

# Prediction button
if st.button("Predict Risk"):
    # Put inputs into dataframe
    data = pd.DataFrame([[loanamount, termdays, age]], 
                        columns=["loanamount", "termdays", "age"])

    # Run model
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    # Show result
    st.subheader(f"Prediction: {'⚠️ Default Risk' if pred==1 else '✅ Good Borrower'}")
    st.write(f"Default Probability: {prob:.2%}")
