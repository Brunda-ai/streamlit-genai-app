from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
import prompts

class IntentRouter:
    def __init__(self, llm):
        self.llm = llm
        
        # Configure the structured parser configuration layer
        self.classifier_parser = JsonOutputParser(pydantic_object=prompts.QueryClassification)
        
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.SYSTEM_ROUTER_INSTRUCTION),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]).partial(format_instructions=self.classifier_parser.get_format_instructions())
        
        self.classifier_chain = self.classifier_prompt | self.llm | self.classifier_parser

    def classify_intent(self, query: str, history: list) -> str:
        """Determines intent profile category from current user input and tracking history."""
        try:
            result = self.classifier_chain.invoke({"input": query, "chat_history": history})
            return result.get("category", "General Knowledge")
        except Exception:
            return "General Knowledge"