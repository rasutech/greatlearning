Log Advisor
A proof-of-concept Streamlit application for analyzing application logs without exposing sensitive information. This tool is designed to help support personnel understand log data without needing direct access to potentially sensitive information.
Features

Transaction ID enrichment via configurable APIs
Log searching in Elasticsearch based on transaction details
LLM-powered log analysis that provides human-readable insights
Clean UI with tabs for analysis, transaction details, and raw logs
Support for multiple applications with configurable log sources

Architecture
The application is designed with a modular, service-based architecture:

EnrichmentService: Handles transaction enrichment via external APIs
LogSearchService: Searches for logs in Elasticsearch
LogAnalysisService: Analyzes logs using LLM (Google Gemini in this example)
ApplicationService: Coordinates the overall workflow

Database Schema
The application uses SQLite for configuration storage with the following tables:

LoggingApps: Stores application details and log sources
LoggingAppFields: Maps transaction IDs to applications
AppEnrichmentAPI: Stores API details for transaction enrichment

Setup Instructions
Prerequisites

Python 3.8+
Elasticsearch (optional for demo)
API key for Google Gemini or alternative LLM

Installation

Clone the repository
Install dependencies:
Copypip install -r requirements.txt

Configure environment variables in .env file:
Copy# Elasticsearch Configuration
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=changeme

# LLM API Keys
GEMINI_API_KEY=your_gemini_api_key_here


Running the Application
Copystreamlit run app.py
Extensibility
The application is designed to be easily extensible:

Adding new applications: Update the LoggingApps table
Supporting new API types: Extend the EnrichmentService class
Using different LLMs: Modify the get_llm_client() function and LogAnalysisService class
Custom log sources: Update the LogSearchService class

Security Considerations

Sensitive data is masked in the enrichment response
LLM is instructed not to include sensitive data in analysis
No raw logs are stored within the application
API credentials are stored in environment variables, not in code

Future Enhancements

Add authentication and user management
Implement role-based access control
Add support for multiple LLM providers
Create a dashboard for common log patterns
Add export functionality for analysis reports
Implement caching for improved performance
