import streamlit as st
from page.single_stock_pred import single_stock_page
from page.portfolio_pred import portfolio_pred_page

# Streamlit App Title
st.title("Enhanced Stock Prediction App")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(" ", ["Single Stock Prediction", "Portfolio Analysis"],
                        captions=[
                                "Predict behaviour for one stock.",
                                "Add your portfolio and predict behaviour for entire portfolio.",
                            ])

if page == "Single Stock Prediction":
    single_stock_page()
elif page == "Portfolio Analysis":
    portfolio_pred_page()
