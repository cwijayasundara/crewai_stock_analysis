import streamlit as st

from crewai import Crew
from stock_analysis_agents import StockAnalysisAgents
from stock_analysis_tasks import StockAnalysisTasks
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")

ticker_map = {
    'Tesla': 'TSLA',
    'Nvidia': 'NVDA',
    'Alphabet': 'GOOGL'
}

options = ["Tesla", "Alphabet", "Nvidia"]


class FinancialCrew:
    def __init__(self, company):
        self.company = company

    def run(self):
        agents = StockAnalysisAgents()
        tasks = StockAnalysisTasks()

        research_analyst_agent = agents.research_analyst()
        financial_analyst_agent = agents.financial_analyst()
        investment_advisor_agent = agents.investment_advisor()

        research_task = tasks.research(research_analyst_agent, self.company)
        financial_task = tasks.financial_analysis(financial_analyst_agent)
        filings_task = tasks.filings_analysis(financial_analyst_agent)
        recommend_task = tasks.recommend(investment_advisor_agent)

        crew = Crew(
            agents=[
                research_analyst_agent,
                financial_analyst_agent,
                investment_advisor_agent
            ],
            tasks=[
                research_task,
                financial_task,
                filings_task,
                recommend_task
            ],
            verbose=True
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":

    st.subheader("AI Agents Conducting Market Research!", divider="rainbow", anchor=False)
    st.image("images/agents.webp", width=400)

    with st.sidebar:
        st.header("👇 Select a company to analyze")
        with st.form("my_form"):
            option = st.selectbox('Select Company ?', options)
            submitted = st.form_submit_button("Submit")
        st.divider()

if submitted:
    with st.status("🤖 **Agents at work...**", state="running", expanded=True) as status:
        with st.container(height=500, border=False):
            trade_crew = FinancialCrew(ticker_map[option])
            result = trade_crew.run()
        status.update(label="✅ Ticker Analysis Ready!", state="complete", expanded=False)
    st.subheader("Here is your market analysis", anchor=False, divider="rainbow")
    st.markdown(result)
