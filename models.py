import re
import ast
import json
from typing import List
from langchain.output_parsers import ListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class PropositionList(BaseModel):
    """Pydantic class representing the type of output"""
    propositions: List[str] = Field(description="List of propositions")


class CustomListParser(ListOutputParser):
    """A concrete subclass of ListOutputParser that implements the 'parse' method."""

    def parse(self, text: str):
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

            Returns:
                A list of strings.
        """
        text = text.strip().strip('``` json\n')
        text = re.sub(r'\\_', '_', text)

        try:
            propositions = json.loads(text)
            if isinstance(propositions, list):
                return propositions
            elif isinstance(propositions, dict):
                return list(propositions.values())[0]
            else:
                print("An exception occured while parsing the LLM response")
                return []
            
        except json.JSONDecodeError:
            try:
                propositions = ast.literal_eval(text)
                if isinstance(propositions, list):
                    return propositions
                else:
                    print("Parsed result is not a list.")
                    return []
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing text as literal: {e}")
                return []
