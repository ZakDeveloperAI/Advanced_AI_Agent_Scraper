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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)