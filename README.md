<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="./images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# SambaNova AI Starter Kits

# Overview

This project is the SambaNova Analyst Assistant, a powerful GenAI-based tool designed to assist businesses in analyzing sales data, generating predictive insights, and creating downloadable reports. Built using the Enterprise Knowledge Retriever Kit, this AI-driven assistant helps users upload and process data from various sources, chat with PDFs, and gain valuable business insights.

The assistant’s user-friendly interface allows users to upload files, create or use an existing vector database, and load or save chat history for seamless, continuous data analysis. It generates personalized, downloadable summaries based on user queries and provides advanced analytics, such as next year’s sales predictions, region-wise sales, and category-wise breakdowns.

Key features include sales insights like hierarchical views of sales, sales vs profit comparisons, and top 10 customer analyses by sales. The AI-driven tool also helps visualize discount vs profit metrics and provides comprehensive sales reports, making it a powerful solution for data-driven decision-making. With its ability to predict future sales trends and generate tailored reports, the SambaNova Analyst Assistant simplifies and enhances the process of business analytics.


# The technical working
LLM Integration: The platform integrates the GPT-2 model via the Hugging Face transformers library, leveraging the SambaNova Enterprise Knowledge Retriever Starter Kit. It generates summaries, sales forecasts, and recommendations based on user inputs such as sales data and categorical information. The platform also supports interactive chat with uploaded PDFs, allowing users to ask questions and receive insights from the PDF content.

Prompt Engineering: Custom prompt templates structure user inputs, ensuring coherent and detailed responses from the GPT-2 model. These templates enable executive summaries, regional breakdowns, and category performance analysis, ensuring high relevance and accuracy in the model’s outputs.

Streamlit Interface: Built using Streamlit, the front end provides an intuitive UI where users can upload CSV files, filter data by various parameters, and visualize trends with interactive charts. It also supports PDF uploads for chat-based interaction. Users can save and load their chat history, ensuring a seamless experience by storing and retrieving previous conversations.

Backend Processing: The backend processes inputs with custom functions that leverage machine learning models like Prophet for forecasting and linear regression for trends analysis. It also performs anomaly detection and customer segmentation, while efficiently handling both CSV and PDF data for intelligent, context-aware responses.

Output: The platform generates comprehensive reports, including executive summaries, regional breakdowns, customer segmentation, anomaly detection, and sales forecasts. Users can download these reports as text files and interact with PDF content via chat. The platform also includes features for saving and loading chat history, making it easy to store and revisit previous conversations.



# Execution the project
To run the project using a virtual environment (either virtualenv or conda), follow these steps in your project terminal:

Navigate to the project directory:
cd ai_starter_kit/enterprise_knowledge_retriever

Create a virtual environment: For virtualenv, run:
python3 -m venv enterprise_knowledge_env

Activate the virtual environment: On macOS or Linux, run:
source enterprise_knowledge_env/bin/activate

On Windows, run:
.\enterprise_knowledge_env\Scripts\activate

Install the required packages:
pip install -r requirements.txt

Run the application by:
streamlit run streamlit/app.py --browser.gatherUsageStats false 

