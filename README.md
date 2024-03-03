## AI Research Agent
This AI Research Agent automates in-depth research on specified topics using advanced NLP models and web scraping. It leverages LangChain, OpenAI's GPT models, and custom scraping tools to search for information, extract content from websites, and generate concise summaries. This project is designed to be deployed as a web service using FastAPI, enabling easy integration with other applications or services, such as automation platforms like make.com.

## Features
Automated Web Searching: Utilizes custom search functionality to find relevant information across the web.
Web Scraping: Extracts content from specified URLs to gather data for research.
Content Summarization: Overcomes token limits by summarizing lengthy content, ensuring the agent can process and understand large documents.
Integration Ready: Designed as a FastAPI application, making it easy to deploy as a web service and integrate with platforms like make.com for automation purposes.
# Prerequisites
Before you can run the AI Research Agent, you need to have the following installed:

Python 3.8 or higher
pip (Python package installer)
## Installation
Clone the Repository

bash
Copy code
git clone <your-repo-link>
cd <your-repo-directory>
## Install Dependencies

Install all the required Python packages using:

Copy code
pip install -r requirements.txt
# Environment Variables
You need to set up the following environment variables in a .env file:

BROWSERLESS_API_KEY: Your API key for browserless.io, used for web scraping.
SERP_API_KEY: Your API key for the SERP API, used for web searching.
Usage
To start the FastAPI server and use the AI Research Agent, run:

css
Copy code
uvicorn main:app --reload
This will start a local server. You can interact with the API by sending POST requests to the / endpoint with a JSON body containing your query, like so:

json
Copy code
{
  "query": "Your research topic here"
}
## Deployment
For deployment on platforms like render.com, refer to the official Render Deployment Documentation and ensure you have set up the necessary environment variables.

## Contributing
Contributions to enhance the AI Research Agent are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.



