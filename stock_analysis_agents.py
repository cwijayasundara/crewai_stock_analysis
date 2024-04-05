import streamlit as st

from crewai import Agent

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4-0125-preview',
                 temperature=0.1)


def streamlit_callback(step_output):
    # This function will be called after each step of the agent's execution
    st.markdown("---")
    for step in step_output:
        if isinstance(step, tuple) and len(step) == 2:
            action, observation = step
            if isinstance(action, dict) and "tool" in action and "tool_input" in action and "log" in action:
                st.markdown(f"# Action")
                st.markdown(f"**Tool:** {action['tool']}")
                st.markdown(f"**Tool Input** {action['tool_input']}")
                st.markdown(f"**Log:** {action['log']}")
                st.markdown(f"**Action:** {action['Action']}")
                st.markdown(
                    f"**Action Input:** ```json\n{action['tool_input']}\n```")
            elif isinstance(action, str):
                st.markdown(f"**Action:** {action}")
            else:
                st.markdown(f"**Action:** {str(action)}")

            st.markdown(f"**Observation**")
            if isinstance(observation, str):
                observation_lines = observation.split('\n')
                for line in observation_lines:
                    if line.startswith('Title: '):
                        st.markdown(f"**Title:** {line[7:]}")
                    elif line.startswith('Link: '):
                        st.markdown(f"**Link:** {line[6:]}")
                    elif line.startswith('Snippet: '):
                        st.markdown(f"**Snippet:** {line[9:]}")
                    elif line.startswith('-'):
                        st.markdown(line)
                    else:
                        st.markdown(line)
            else:
                st.markdown(str(observation))
        else:
            st.markdown(step)


class StockAnalysisAgents:
    def financial_analyst(self):
        return Agent(
            role='The Best Financial Analyst',
            goal="""Impress all customers with your financial data and market trends analysis""",
            backstory="""The most seasoned financial analyst with lots of expertise in stock market analysis and 
            investment  strategies that is working for a super important customer.""",
            verbose=True,
            step_callback=streamlit_callback,
            llm=llm,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                CalculatorTools.calculate,
                SECTools.search_10q,
                SECTools.search_10k
            ]
        )

    def research_analyst(self):
        return Agent(
            role='Staff Research Analyst',
            goal="""Being the best at gather, interpret data and amaze your customer with it""",
            backstory="""Known as the BEST research analyst, you're  skilled in sifting through news, company 
            announcements, and market sentiments. Now you're working on a super  important customer""",
            verbose=True,
            step_callback=streamlit_callback,
            llm=llm,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                SearchTools.search_news,
                YahooFinanceNewsTool(),
                SECTools.search_10q,
                SECTools.search_10k
            ]
        )

    def investment_advisor(self):
        return Agent(
            role='Private Investment Advisor',
            goal="""Impress your customers with full analyses over stocks and completer investment recommendations""",
            backstory="""You're the most experienced investment advisor and you combine various analytical insights 
            to formulate strategic investment advice. You are now working for a super important customer you need to 
            impress.""",
            verbose=True,
            step_callback=streamlit_callback,
            llm=llm,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                SearchTools.search_news,
                CalculatorTools.calculate,
                YahooFinanceNewsTool()
            ]
        )
