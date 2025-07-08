from typing import Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage
from .models import CompanyAnalysis, CompanyInfo, ResearchState
from .firecrawl import FirecrawlService
from .prompts import DeveloperToolsPrompts
from dotenv import load_dotenv

load_dotenv()

class Workflow:
    def __init__(self):
        self.firecrawl = FirecrawlService()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        self.prompts = DeveloperToolsPrompts()
        self.workflow = self._build_workflow()
        
    def _build_workflow(self):
        pass
    
    def _extract_tools_step(self,state: ResearchState)->Dict[str, Any]:
        print(f"Finding articles about {state.query}...")
        
        article_query = f"{state.query} tools comparison best alternatives"
        search_results = self.firecrawl.search_companies(article_query, num_results=3)
        
        all_content=""
        for result in search_results.data:
            url=result.get("url","")
            scraped=self.firecrawl.scrape_company_pages(url)
            if scraped:
                all_content += scraped.markdown[:1500] + "\n\n"
        
        messages=[
            SystemMessage(content=self.prompts.TOOL_EXTRACTION_SYSTEM),
            HumanMessage(content=self.prompts.tool_extraction_user(state.query))
        ]
        
        try:
            response=self.llm.ainvoke(messages)
            tool_names=[
                name.strip()
                for name in response.content.strip().split("\n")
                if name.strip()
            ]
            print(f"Extracted tools: {', '.join(tool_names[:5])}")
            return {"extracted_tools": tool_names}
        except Exception as e:
            print(f"Error during tool extraction: {e}")
            return {"extracted_tools": []}