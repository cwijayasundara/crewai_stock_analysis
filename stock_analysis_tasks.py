from crewai import Task
from textwrap import dedent
import datetime


class StockAnalysisTasks():
    def research(self, agent, company):
        return Task(description=dedent(f""" Collect and summarize recent news articles, press releases, and market 
        analyses related to the stock and  its industry.
        
        Pay special attention to any significant events, market sentiments, and analysts' opinions. Also include 
        upcoming  events like earnings and others.
  
        Your final answer MUST be a report that includes a comprehensive summary of the latest news, any notable  
        shifts in market sentiment, and potential impacts on  the stock.
       
        Also make sure to return the stock ticker.
        
        {self.__tip_section()}
  
        Make sure to use the most recent data as possible. Selected company by the customer: {company}  """),
                    agent=agent)

    def financial_analysis(self, agent):
        return Task(description=dedent(f"""
        Conduct a thorough analysis of the stock's financial  health and market performance.
        
        This includes examining key financial metrics such as P/E ratio, EPS growth, revenue trends, 
        and  debt-to-equity ratio.
         
        Also, analyze the stock's performance in comparison to its industry peers and overall market trends.
        
        Your final report MUST expand on the summary provided but now including a clear assessment of the stock's 
        financial standing, its strengths and weaknesses, and how it fares against its competitors in the current  
        market scenario.{self.__tip_section()}

        Make sure to use the most recent data possible.  """), agent=agent)

    def filings_analysis(self, agent):
        return Task(description=dedent(f"""
        Analyze the latest 10-Q and 10-K filings from EDGAR for  the stock in question.
       
        Focus on key sections like Management's Discussion and Analysis, financial statements, insider trading 
        activity, and any disclosed risks.
        
        Extract relevant data and insights that could influence  the stock's future performance.
       
        Your final answer must be an expanded report that now also highlights significant findings from these 
        filings, including any red flags or positive indicators for your customer. {self.__tip_section()}  """),
                    agent=agent)

    def recommend(self, agent):
        return Task(description=dedent(f"""
        Review and synthesize the analyses provided by the Financial Analyst and the Research Analyst.
        
        Combine these insights to form a comprehensive investment recommendation.
       
        You MUST Consider all aspects, including financial health, market sentiment, and qualitative data from  EDGAR 
        filings.

        Make sure to include a section that shows insider  trading activity, and upcoming events like earnings.

        Your final answer MUST be a recommendation for your customer. It should be a full super detailed report, 
        providing a clear investment stance and strategy with supporting evidence.
        
        Make it pretty and well formatted for your customer. Please follow the template below:
        
        Investment opportunity:
        
        - Value proposition.
        - Unique selling points.
        - Potential returns:
            - Expected ROI.
            - Risk-adjusted returns.
        - Risks and Mitigation Strategies:
        - Identification of potential risks:
            - Market risks, operational risks, regulatory risks, etc.
        - Mitigation strategies:
            - Plans to address identified risks.
        - Strategic Growth Initiatives:
        Expansion plans:
            - Geographic expansion, product diversification, etc.
        Research and development:
            - Innovation pipeline and investments.
        Strategic partnerships:
            - Alliances, joint ventures, collaborations.
        Valuation:
            Valuation methodology:
                - Discounted Cash Flow (DCF), Comparable Company Analysis (CCA), etc.
                - Valuation assumptions and inputs.
        Investment Recommendations:
            - Summary of key findings and analysis.
        Investment decision:
            - Buy, sell, hold recommendations.
            - Justification of recommendations.
        Conclusion:
            - Summary of the investment opportunity.
            - Closing remarks.
        Appendix:
        - Detailed financial tables.
        - Glossary of financial terms.
        - Assumptions used in the analysis.
        References:
        - Sources of information used in the report.
        Today's date: 
        {self.__tip_section()}  """) + str(datetime.date.today()), agent=agent)

    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"
