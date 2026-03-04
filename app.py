# ================================
# IMPORTS
# ================================
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Tourism Package Analytics & Prediction",
    layout="wide"
)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("travel_dataset.csv")
    return df

df = load_data()

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "Travel_package.pkl"
    )
    model = joblib.load(model_path)
    return model

model = load_model()

# ================================
# SIDEBAR NAVIGATION
# ================================
st.sidebar.title("🌍 Tourism Analytics Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "About Dataset",
        "Univariate Analysis",
        "Bivariate Analysis",
        "Model Prediction"
    ]
)

# ================================
# KPI SECTION
# ================================
if page != "Model Prediction":

    st.title("Tourism Customer Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", len(df))

    with col2:
        purchase_rate = df["ProdTaken"].mean()*100
        st.metric("Purchase Rate", f"{purchase_rate:.2f}%")

    with col3:
        avg_income = df["MonthlyIncome"].mean()
        st.metric("Average Income", f"{avg_income:,.0f}")

    st.divider()

# ================================
# ABOUT DATASET
# ================================
if page == "About Dataset":

    st.header("About the Dataset")

    st.write("""
This dataset belongs to the **tourism domain** and contains customer demographic,
travel behavior, and marketing interaction data.

The objective of this project is to **predict whether a customer will purchase a tourism package**.

The target variable is **ProdTaken**:
- 1 → Customer purchased package
- 0 → Customer did not purchase
""")

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.write("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Feature Description")

    st.write("""
Key features include:

- Age – Customer age
- CityTier – City development level
- Occupation – Customer occupation
- MonthlyIncome – Customer monthly income
- NumberOfTrips – Trips per year
- Passport – Passport ownership
- OwnCar – Car ownership
- ProductPitched – Package offered to customer
""")

# ================================
# UNIVARIATE ANALYSIS
# ================================
elif page == "Univariate Analysis":

    st.header("Univariate Analysis")

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns

    feature_type = st.selectbox(
        "Select Feature Type",
        ["Numeric","Categorical"]
    )

    if feature_type == "Numeric":

        col = st.selectbox("Select Numeric Feature", numeric_cols)

        fig = px.histogram(
            df,
            x=col,
            nbins=40,
            title=f"Distribution of {col}"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        col = st.selectbox("Select Categorical Feature", categorical_cols)

        fig = px.bar(
            df[col].value_counts(),
            title=f"Distribution of {col}"
        )

        st.plotly_chart(fig, use_container_width=True)

# ================================
# BIVARIATE ANALYSIS
# ================================
elif page == "Bivariate Analysis":

    st.header("Bivariate Analysis")

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Scatter Plot","Box Plot"]
    )

    if chart_type == "Scatter Plot":

        x = st.selectbox("X Axis", numeric_cols)
        y = st.selectbox("Y Axis", numeric_cols, index=1)

        fig = px.scatter(
            df,
            x=x,
            y=y,
            color="ProdTaken",
            title=f"{x} vs {y}"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        cat = st.selectbox("Categorical Feature", categorical_cols)
        num = st.selectbox("Numeric Feature", numeric_cols)

        fig = px.box(
            df,
            x=cat,
            y=num,
            color=cat,
            title=f"{num} by {cat}"
        )

        st.plotly_chart(fig, use_container_width=True)

# ================================
# MODEL PREDICTION
# ================================
elif page == "Model Prediction":

    st.title("Tourism Package Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 18, 80, 30)

        type_contact = st.selectbox(
            "Type of Contact",
            ["Self Inquiry", "Company Invited"]
        )

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        duration_pitch = st.slider(
            "Duration Of Pitch (minutes)",
            0, 60, 15
        )

        occupation = st.selectbox(
            "Occupation",
            df["Occupation"].unique()
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        persons = st.number_input(
            "Number of Persons Visiting",
            1, 10, 2
        )

        followups = st.slider(
            "Number of Followups",
            0, 10, 2
        )

        children = st.number_input(
            "Number Of Children Visiting",
            0, 5, 0
        )

    with col2:

        product = st.selectbox(
            "Product Pitched",
            df["ProductPitched"].unique()
        )

        property_star = st.selectbox(
            "Preferred Property Star",
            [3, 4, 5]
        )

        marital = st.selectbox(
            "Marital Status",
            df["MaritalStatus"].unique()
        )

        trips = st.slider(
            "Number Of Trips Per Year",
            0, 10, 2
        )

        passport = st.selectbox(
            "Passport",
            [0, 1]
        )

        pitch_score = st.slider(
            "Pitch Satisfaction Score",
            1, 5, 3
        )

        car = st.selectbox(
            "Own Car",
            [0, 1]
        )

        designation = st.selectbox(
            "Designation",
            df["Designation"].unique()
        )

        income = st.number_input(
            "Monthly Income",
            1000, 20000000, 30000
        )

    predict = st.button("Predict")

    if predict:

        input_df = pd.DataFrame({
            "Age":[age],
            "TypeofContact":[type_contact],
            "CityTier":[city_tier],
            "DurationOfPitch":[duration_pitch],
            "Occupation":[occupation],
            "Gender":[gender],
            "NumberOfPersonVisiting":[persons],
            "NumberOfFollowups":[followups],
            "ProductPitched":[product],
            "PreferredPropertyStar":[property_star],
            "MaritalStatus":[marital],
            "NumberOfTrips":[trips],
            "Passport":[passport],
            "PitchSatisfactionScore":[pitch_score],
            "OwnCar":[car],
            "NumberOfChildrenVisiting":[children],
            "Designation":[designation],
            "MonthlyIncome":[income]
        })

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.metric("Purchase Probability", f"{prob*100:.2f}%")

        threshold = 0.35

        if prob > threshold:
            st.success("Customer is likely to purchase the tourism package")
        else:
            st.error("Customer is unlikely to purchase the tourism package")