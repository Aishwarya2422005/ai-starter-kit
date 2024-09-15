import os
import sys
import logging
import yaml
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import base64
from prophet import Prophet

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval
from utils.visual.env_utils import env_input_fields, initialize_env_variables, are_credentials_set, save_credentials
from utils.vectordb.vector_db import VectorDb

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, f"data/my-vector-db")

logging.basicConfig(level=logging.INFO)
logging.info("URL: http://localhost:8501")


def predict_next_year_sales(df):
    # Prepare the data for Prophet
    df_prophet = df.groupby('Ship Date')['Sales'].sum().reset_index()
    df_prophet.columns = ['ds', 'y']

    # Create and fit the model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_prophet)

    # Make future dataframe for predictions (next year)
    last_date = df['Ship Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365)
    future_df = pd.DataFrame({'ds': future_dates})

    # Make predictions
    forecast = model.predict(future_df)

    return forecast


def generate_download_link(report_content):
    b64 = base64.b64encode(report_content.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="comprehensive_sales_report.txt">Download Comprehensive Report</a>'

def save_chat_history():
    history_data = {
        "chat_history": st.session_state.chat_history,
        "sources_history": st.session_state.sources_history
    }

    with open("chat_history.json", "w") as f:
        json.dump(history_data, f)

    st.success("Chat history saved successfully!")


def load_chat_history():
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r") as f:
            history_data = json.load(f)

        st.session_state.chat_history = history_data["chat_history"]
        st.session_state.sources_history = history_data["sources_history"]
        st.success("Chat history loaded successfully!")
    else:
        st.warning("No saved chat history found.")


def load_csv(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['Ship Date'] = pd.to_datetime(df['Ship Date'])
            return df
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
    return None


def handle_userinput(user_question):
    if user_question:
        try:
            with st.spinner("Processing..."):
                response = st.session_state.conversation.invoke({"question": user_question})
            st.session_state.chat_history.append(user_question)
            st.session_state.chat_history.append(response["answer"])

            sources = set([
                f'{sd.metadata["filename"]}'
                for sd in response["source_documents"]
            ])
            sources_text = ""
            for index, source in enumerate(sources, start=1):
                source_link = source
                sources_text += (
                    f'<font size="2" color="grey">{index}. {source_link}</font>  \n'
                )
            st.session_state.sources_history.append(sources_text)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")

    for ques, ans, source in zip(
            st.session_state.chat_history[::2],
            st.session_state.chat_history[1::2],
            st.session_state.sources_history,
    ):
        with st.chat_message("user"):
            st.write(f"{ques}")

        with st.chat_message(
                "ai",
                avatar="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
        ):
            st.write(f"{ans}")
            if st.session_state.show_sources:
                with st.expander("Sources"):
                    st.markdown(
                        f'<font size="2" color="grey">{source}</font>',
                        unsafe_allow_html=True,
                    )


def initialize_document_retrieval():
    if are_credentials_set():
        try:
            return DocumentRetrieval()
        except Exception as e:
            st.error(f"Failed to initialize DocumentRetrieval: {str(e)}")
            return None
    return None


def generate_executive_summary(df):
    total_sales = df['Sales'].sum()
    previous_year_sales = df[df['Ship Date'].dt.year == df['Ship Date'].dt.year.min()]['Sales'].sum()
    sales_growth = (total_sales - previous_year_sales) / previous_year_sales * 100
    top_region = df.groupby('Region')['Sales'].sum().idxmax()
    top_category = df.groupby('Category')['Sales'].sum().idxmax()

    summarizer = pipeline("text-generation", model="gpt2")
    prompt = f"Total sales for the fiscal year {df['Ship Date'].dt.year.max()} reached ${total_sales:,.0f}, "
    prompt += f"showing a {sales_growth:.1f}% increase compared to the previous year. "
    prompt += f"The highest-performing region was {top_region}, "
    prompt += f"while {top_category} was the best-selling category. "
    prompt += "Based on this data, a business analyst might conclude that:"

    summary = summarizer(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
    return summary


def generate_regional_breakdown(df):
    region_sales = df.groupby(['Region', df['Ship Date'].dt.year])['Sales'].sum().unstack()
    region_growth = (region_sales[region_sales.columns[-1]] - region_sales[region_sales.columns[-2]]) / region_sales[
        region_sales.columns[-2]] * 100

    summarizer = pipeline("text-generation", model="gpt2")
    prompt = "Regional sales breakdown:\n"
    for region in region_sales.index:
        prompt += f"{region} region: ${region_sales.loc[region, region_sales.columns[-1]]:,.0f} "
        prompt += f"({region_growth[region]:.1f}% growth). "
    prompt += "Analyzing this data, we can conclude:"

    breakdown = summarizer(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
    return breakdown


def generate_category_performance(df):
    category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)

    summarizer = pipeline("text-generation", model="gpt2")
    prompt = "Category-wise sales performance:\n"
    for category, sales in category_sales.items():
        prompt += f"{category}: ${sales:,.0f}. "
    prompt += "Based on these figures, we can infer:"

    performance = summarizer(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
    return performance


def generate_monthly_trends(df):
    monthly_sales = df.groupby(pd.Grouper(key='Ship Date', freq='M'))['Sales'].sum()
    peak_month = monthly_sales.idxmax().strftime('%B %Y')
    lowest_month = monthly_sales.idxmin().strftime('%B %Y')

    summarizer = pipeline("text-generation", model="gpt2")
    prompt = f"Monthly sales analysis shows that {peak_month} had the highest sales, "
    prompt += f"while {lowest_month} saw the lowest figures. "
    prompt += "Considering these trends, a sales analyst might conclude:"

    trends = summarizer(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
    return trends


def generate_customer_segmentation(df):
    segment_sales = df.groupby('Segment')['Sales'].sum()
    total_sales = segment_sales.sum()

    summarizer = pipeline("text-generation", model="gpt2")
    prompt = "Customer segmentation insights:\n"
    for segment, sales in segment_sales.items():
        percentage = (sales / total_sales) * 100
        prompt += f"{segment}: ${sales:,.0f} ({percentage:.1f}% of total sales). "
    prompt += "Analyzing this segmentation, we can deduce:"

    segmentation = summarizer(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
    return segmentation


def generate_sales_forecast(df):
    df['Days'] = (df['Ship Date'] - df['Ship Date'].min()).dt.days
    X = df[['Days']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    last_date = df['Ship Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365)
    future_X = pd.DataFrame({'Days': (future_dates - df['Ship Date'].min()).days})
    future_sales = model.predict(future_X)

    summarizer = pipeline("text-generation", model="gpt2")
    prompt = f"Based on our sales forecast model, "
    prompt += f"sales are projected to reach ${future_sales.sum():,.0f} in the next year. "
    prompt += "Considering this projection, a business strategist might recommend:"

    forecast = summarizer(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
    return forecast


def generate_anomaly_detection(df):
    # Simple anomaly detection based on Z-score
    df['Sales_ZScore'] = (df['Sales'] - df['Sales'].mean()) / df['Sales'].std()
    anomalies = df[df['Sales_ZScore'].abs() > 3]

    summarizer = pipeline("text-generation", model="gpt2")
    prompt = f"Anomaly detection identified {len(anomalies)} potential outliers in the sales data. "
    if not anomalies.empty:
        prompt += f"The most significant anomaly occurred on {anomalies['Ship Date'].dt.date.iloc[0]}, "
        prompt += f"with a sale of ${anomalies['Sales'].iloc[0]:,.2f}. "
    prompt += "Based on these anomalies, we should consider:"

    anomaly_insights = summarizer(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
    return anomaly_insights


def generate_recommendations(df):
    summarizer = pipeline("text-generation", model="gpt2")
    prompt = "Based on the overall sales data and trends, here are some recommendations:\n"
    prompt += "1. Focus on expanding sales in the top-performing region.\n"
    prompt += "2. Investigate and address any issues in the lowest-performing category.\n"
    prompt += "3. Prepare for peak sales periods identified in the monthly trends.\n"
    prompt += "4. Develop targeted strategies for each customer segment.\n"
    prompt += "5. Address any anomalies detected in the sales data.\n"
    prompt += "Expanding on these recommendations, we suggest:"

    recommendations = summarizer(prompt, max_length=400, num_return_sequences=1)[0]['generated_text']
    return recommendations



def main():
    os.environ['SAMBANOVA_API_KEY'] = '1c4ab220-42ac-4812-8c30-7233e794c266'
    with open(CONFIG_PATH, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    prod_mode = config.get('prod_mode', False)
    default_collection = 'ekr_default_collection'

    initialize_env_variables(prod_mode)

    st.set_page_config(
        page_title="SambaNova Analyst Assistant",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
        layout="wide"
    )
    for var in ['conversation', 'chat_history', 'show_sources', 'sources_history', 'vectorstore', 'input_disabled',
                'document_retrieval', 'df']:
        if var not in st.session_state:
            st.session_state[var] = None if var != 'chat_history' and var != 'sources_history' else []

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    #if 'input_disabled' not in st.session_state:
    #   st.session_state.input_disabled = False
    if 'document_retrieval' not in st.session_state:
        st.session_state.document_retrieval = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
    if 'input_disabled' not in st.session_state or st.session_state.input_disabled is None:
        st.session_state.input_disabled = False

    st.title(":orange[SambaNova] Analyst Assistant")
    st.write(f"Input disabled: {st.session_state.input_disabled}")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save Chat History"):
            save_chat_history()
    with col2:
        if st.button("Load Chat History"):
            load_chat_history()
    with col3:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.sources_history = []
            st.success("Chat history cleared!")

    with st.sidebar:
        st.title("Setup")

        if not are_credentials_set():
            url, api_key = env_input_fields()
            if st.button("Save Credentials", key="save_credentials_sidebar"):
                message = save_credentials(url, api_key, prod_mode)
                st.success(message)
                st.rerun()
        else:
            st.success("Credentials are set")
            if st.button("Clear Credentials", key="clear_credentials"):
                save_credentials("", "", prod_mode)
                st.rerun()

        if are_credentials_set():
            if st.session_state.document_retrieval is None:
                st.session_state.document_retrieval = initialize_document_retrieval()

        if st.session_state.document_retrieval is not None:
            st.markdown("**1. Pick a datasource**")

            datasource_options = ["Upload CSV", "Upload files (create new vector db)"]
            if not prod_mode:
                datasource_options.append("Use existing vector db")

            datasource = st.selectbox("", datasource_options)

            if "Upload CSV" in datasource:
                uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
                if uploaded_file:
                    df = load_csv(uploaded_file)
                    if df is not None:
                        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                        st.session_state.df = df
                        st.session_state.input_disabled = False

            elif "Upload files" in datasource:
                if config.get('pdf_only_mode', False):
                    docs = st.file_uploader(
                        "Add PDF files", accept_multiple_files=True, type=["pdf"]
                    )
                else:
                    docs = st.file_uploader(
                        "Add files", accept_multiple_files=True,
                        type=[".eml", ".html", ".json", ".md", ".msg", ".rst", ".rtf", ".txt", ".xml", ".png", ".jpg",
                              ".jpeg", ".tiff", ".bmp", ".heic", ".csv", ".doc", ".docx", ".epub", ".odt", ".pdf",
                              ".ppt", ".pptx", ".tsv", ".xlsx"]
                    )
                st.markdown("**2. Process your documents and create vector store**")
                st.markdown(
                    "**Note:** Depending on the size and number of your documents, this could take several minutes"
                )
                st.markdown("Create database")
                if st.button("Process"):
                    with st.spinner("Processing"):
                        try:
                            text_chunks = st.session_state.document_retrieval.parse_doc(docs)
                            embeddings = st.session_state.document_retrieval.load_embedding_model()
                            collection_name = default_collection if not prod_mode else None
                            vectorstore = st.session_state.document_retrieval.create_vector_store(text_chunks,
                                                                                                  embeddings,
                                                                                                  output_db=None,
                                                                                                  collection_name=collection_name)
                            st.session_state.vectorstore = vectorstore
                            st.session_state.document_retrieval.init_retriever(vectorstore)
                            st.session_state.conversation = st.session_state.document_retrieval.get_qa_retrieval_chain()
                            st.toast(f"File uploaded! Go ahead and ask some questions", icon='🎉')
                            st.session_state.input_disabled = False
                        except Exception as e:
                            st.error(f"An error occurred while processing: {str(e)}")

                if not prod_mode:
                    st.markdown("[Optional] Save database for reuse")
                    save_location = st.text_input("Save location", "./data/my-vector-db").strip()
                    if st.button("Process and Save database"):
                        with st.spinner("Processing"):
                            try:
                                text_chunks = st.session_state.document_retrieval.parse_doc(docs)
                                embeddings = st.session_state.document_retrieval.load_embedding_model()
                                vectorstore = st.session_state.document_retrieval.create_vector_store(text_chunks,
                                                                                                      embeddings,
                                                                                                      output_db=save_location,
                                                                                                      collection_name=default_collection)
                                st.session_state.vectorstore = vectorstore
                                st.session_state.document_retrieval.init_retriever(vectorstore)
                                st.session_state.conversation = st.session_state.document_retrieval.get_qa_retrieval_chain()
                                st.toast(
                                    f"File uploaded and saved to {save_location} with collection '{default_collection}'! Go ahead and ask some questions",
                                    icon='🎉')
                                st.session_state.input_disabled = False
                            except Exception as e:
                                st.error(f"An error occurred while processing and saving: {str(e)}")

            elif not prod_mode and "Use existing" in datasource:
                db_path = st.text_input(
                    f"Absolute path to your DB folder",
                    placeholder="E.g., /Users/<username>/path/to/your/vectordb",
                ).strip()
                st.markdown("**2. Load your datasource and create vectorstore**")
                st.markdown(
                    "**Note:** Depending on the size of your vector database, this could take a few seconds"
                )
                if st.button("Load"):
                    with st.spinner("Loading vector DB..."):
                        if db_path == "":
                            st.error("You must provide a path", icon="🚨")
                        else:
                            if os.path.exists(db_path):
                                try:
                                    embeddings = st.session_state.document_retrieval.load_embedding_model()
                                    collection_name = default_collection if not prod_mode else None
                                    vectorstore = st.session_state.document_retrieval.load_vdb(db_path, embeddings,
                                                                                               collection_name=collection_name)
                                    st.toast(
                                        f"Database loaded{'with collection ' + default_collection if not prod_mode else ''}")
                                    st.session_state.vectorstore = vectorstore
                                    st.session_state.document_retrieval.init_retriever(vectorstore)
                                    st.session_state.conversation = st.session_state.document_retrieval.get_qa_retrieval_chain()
                                    st.session_state.input_disabled = False
                                except Exception as e:
                                    st.error(f"An error occurred while loading the database: {str(e)}")
                            else:
                                st.error("Database not present at " + db_path, icon="🚨")
            st.markdown("**3. Ask questions about your data!**")

            with st.expander("Additional settings", expanded=True):
                st.markdown("**Interaction options**")
                st.markdown(
                    "**Note:** Toggle these at any time to change your interaction experience"
                )
                st.checkbox("Show sources", key="show_sources")


                st.markdown("**Reset chat**")
                st.markdown(
                    "**Note:** Resetting the chat will clear all conversation history"
                )
                if st.button("Reset conversation"):
                    st.session_state.chat_history = []
                    st.session_state.sources_history = []
                    st.toast(
                        "Conversation reset. The next response will clear the history on the screen"
                    )

    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = pd.to_datetime(st.date_input("Start Date", df["Ship Date"].min()))
        with col2:
            end_date = pd.to_datetime(st.date_input("End Date", df["Ship Date"].max()))
        df_filtered = df[(df["Ship Date"] >= start_date) & (df["Ship Date"] <= end_date)]

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            region = st.multiselect("Select Region", df_filtered["Region"].unique())
        with col2:
            category = st.multiselect("Select Category", df_filtered["Category"].unique())
        with col3:
            segment = st.multiselect("Select Segment", df_filtered["Segment"].unique())

        if region:
            df_filtered = df_filtered[df_filtered["Region"].isin(region)]
        if category:
            df_filtered = df_filtered[df_filtered["Category"].isin(category)]
        if segment:
            df_filtered = df_filtered[df_filtered["Segment"].isin(segment)]

        # Visualizations
        col1,col2 = st.columns(2)

        with col1:
            st.subheader("Category-wise Sales")
            category_sales = df_filtered.groupby("Category")["Sales"].sum().sort_values(ascending=False)
            fig = px.bar(category_sales, x=category_sales.index, y="Sales", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Region-wise Sales")
            fig = px.pie(df_filtered, values="Sales", names="Region", hole=0.5)
            st.plotly_chart(fig, use_container_width=True)

        # Time Series Analysis
        st.subheader("Sales Over Time")
        sales_over_time = df_filtered.groupby("Ship Date")["Sales"].sum().reset_index()
        fig = px.line(sales_over_time, x="Ship Date", y="Sales")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Next Year Sales Prediction")
        if st.button("Generate Next Year Sales Forecast"):
            with st.spinner("Generating sales forecast for next year..."):
                forecast = predict_next_year_sales(df)

                # Aggregate predictions by month
                forecast['Month'] = forecast['ds'].dt.to_period('M')
                monthly_forecast = forecast.groupby('Month').agg({
                    'yhat': 'sum',
                    'yhat_lower': 'sum',
                    'yhat_upper': 'sum'
                }).reset_index()
                monthly_forecast['Month'] = monthly_forecast['Month'].dt.to_timestamp()

                # Plot the monthly forecast
                fig = go.Figure()
                fig.add_trace(go.Bar(x=monthly_forecast['Month'], y=monthly_forecast['yhat'],
                                     name='Predicted Sales'))
                fig.add_trace(go.Scatter(x=monthly_forecast['Month'], y=monthly_forecast['yhat_upper'],
                                         mode='lines', name='Upper Bound', line=dict(color='rgba(0,100,80,0.2)')))
                fig.add_trace(go.Scatter(x=monthly_forecast['Month'], y=monthly_forecast['yhat_lower'],
                                         mode='lines', name='Lower Bound', line=dict(color='rgba(0,100,80,0.2)')))
                fig.add_trace(go.Scatter(x=monthly_forecast['Month'], y=monthly_forecast['yhat_lower'],
                                         fill='tonexty', fillcolor='rgba(0,100,80,0.1)',
                                         line=dict(color='rgba(255,255,255,0)'),
                                         name='Confidence Interval'))
                fig.update_layout(title='Next Year Monthly Sales Forecast', xaxis_title='Month',
                                  yaxis_title='Predicted Sales')
                st.plotly_chart(fig, use_container_width=True)

                # Display forecast summary
                st.subheader("Next Year Forecast Summary")
                total_forecasted_sales = forecast['yhat'].sum()
                avg_monthly_sales = monthly_forecast['yhat'].mean()
                highest_month = monthly_forecast.loc[monthly_forecast['yhat'].idxmax()]
                lowest_month = monthly_forecast.loc[monthly_forecast['yhat'].idxmin()]

                st.write(f"Total forecasted sales for next year: ${total_forecasted_sales:,.2f}")
                st.write(f"Average monthly forecasted sales: ${avg_monthly_sales:,.2f}")
                st.write(
                    f"Highest sales month: {highest_month['Month'].strftime('%B %Y')} (${highest_month['yhat']:,.2f})")
                st.write(
                    f"Lowest sales month: {lowest_month['Month'].strftime('%B %Y')} (${lowest_month['yhat']:,.2f})")

                # Calculate year-over-year growth
                current_year_sales = df[df['Ship Date'] >= (df['Ship Date'].max() - timedelta(days=365))]['Sales'].sum()
                yoy_growth = (total_forecasted_sales - current_year_sales) / current_year_sales * 100
                st.write(f"Projected year-over-year growth: {yoy_growth:.2f}%")

        # Treemap
        st.subheader("Hierarchical view of Sales")
        fig = px.treemap(df_filtered, path=["Region", "Category", "Sub-Category"], values="Sales", color="Profit")
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot
        st.subheader("Sales vs Profit")
        fig = px.scatter(df_filtered, x="Sales", y="Profit", size="Quantity", color="Category", hover_name="Product Name")
        st.plotly_chart(fig, use_container_width=True)

        # Top Customers
        st.subheader("Top 10 Customers by Sales")
        top_customers = df_filtered.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(top_customers, x=top_customers.index, y="Sales", text_auto=True)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        # Discount vs Profit
        st.subheader("Discount vs Profit")
        fig = px.scatter(df_filtered, x="Discount", y="Profit", color="Category", size="Sales")
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("Sample Data")
        st.dataframe(df_filtered.head(10))


        st.subheader("Comprehensive Sales Report")
        if st.button("Generate Comprehensive Report"):
            with st.spinner("Generating comprehensive report..."):
                exec_summary = generate_executive_summary(df_filtered)
                regional_breakdown = generate_regional_breakdown(df_filtered)
                category_performance = generate_category_performance(df_filtered)
                monthly_trends = generate_monthly_trends(df_filtered)
                customer_segmentation = generate_customer_segmentation(df_filtered)
                sales_forecast = generate_sales_forecast(df_filtered)
                anomaly_detection = generate_anomaly_detection(df_filtered)
                recommendations = generate_recommendations(df_filtered)

                report_content = f"""Comprehensive Sales Report

                1. Executive Summary
                {exec_summary}

                2. Regional Sales Breakdown
                {regional_breakdown}

                3. Category Performance
                {category_performance}

                4. Monthly Sales Trends
                {monthly_trends}

                5. Customer Segmentation Insights
                {customer_segmentation}

                6. Sales Forecast
                {sales_forecast}

                7. Anomaly Detection
                {anomaly_detection}

                8. Recommendations
                {recommendations}
                """

                st.subheader("1. Executive Summary")
                st.write(exec_summary)

                st.subheader("2. Regional Sales Breakdown")
                st.write(regional_breakdown)

                st.subheader("3. Category Performance")
                st.write(category_performance)

                st.subheader("4. Monthly Sales Trends")
                st.write(monthly_trends)

                st.subheader("5. Customer Segmentation Insights")
                st.write(customer_segmentation)

                st.subheader("6. Sales Forecast")
                st.write(sales_forecast)

                st.subheader("7. Anomaly Detection")
                st.write(anomaly_detection)

                st.subheader("8. Recommendations")
                st.write(recommendations)

                st.markdown(generate_download_link(report_content), unsafe_allow_html=True)

    user_question = st.chat_input("Ask questions about your data", disabled=st.session_state.input_disabled)
    handle_userinput(user_question)

if __name__ == "__main__":
    main()