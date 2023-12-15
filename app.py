import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st

load_dotenv()

browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")


# Tool for search
def search(query):

    url = "https://google.serper.dev/search"

    payload = json.dumps({
    "q": query
    })
    headers = {
    'X-API-KEY': serper_api_key,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    return data

# Tool for scraping 
def scrape_website(objective:str,url:str):
    print("Scraping Website")

    headers={
        'Cache-Control':'no-cache',
        'Content-Type':'application/json'
    }
    #Define the data to be sent in the request
    data ={
        "url":url
    }
    data_json = json.dumps(data)

    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url,headers=headers,data=data_json)

    if response.status_code ==200:
        soup = BeautifulSoup(response.content,"html.parser")
        text = soup.get_text()
        print("Content____",text)
        return text
    else:
        print(f" HTTP request failed with code{response.status_code}")

    if len(text) > 10000:
        output = summmary(objective,text)
        return output
    else:
        return text
    

# summarizing text becuase of token limit barrier
def summmary(objective,content):
    llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-16k")    
    text_splitter = RecursiveCharacterTextSplitter(separator="\n",chunk_size=3000, length_function=len )
    docs = text_splitter.split_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}
    SUMMARY: 
    """

    map_prompt_template =PromptTemplate(template=map_prompt,input_variables=["text","objective"])

    summmary_chain = load_summarize_chain(
        llm=llm,
        chain_type = 'map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose =True
    )
    output = summmary_chain.run(input_documents =docs,objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")




# Create langchain agent with tools share
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)
# def main():
#     st.set_page_config(page_title="AI research agent", page_icon=":bird:")

#     st.header("AI research agent :bird:")
#     query = st.text_input("Research goal")

#     if query:
#         st.write("Doing research for ", query)

#         result = agent({"input": query})

#         st.info(result['output'])


# if __name__ == '__main__':
#     main()

app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content






