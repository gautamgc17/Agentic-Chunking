from tqdm import tqdm
from typing import List
from langchain_ibm import ChatWatsonx 
from langchain.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import OutputFixingParser
from langchain.callbacks.tracers import ConsoleCallbackHandler
from models import CustomListParser
from prompts import *


class CreatePropositions:
    def __init__(self, llm: ChatWatsonx, source_language: str = "English", verbose: bool = True):
        
        self.watsonx_llm = llm
        self.source_language = source_language
        self.verbose = verbose

        
        if self.watsonx_llm.model_id in ["mistralai/mixtral-8x7b-instruct-v01", "mistralai/mistral-large"]:
            self.create_propositions_system_prompt = "[INST]<<SYS>>" + CREATE_PROPOSITIONS_SYSTEM_PROMPT
            self.create_propositions_user_prompt = CREATE_PROPOSITIONS_USER_PROMPT + "[/INST]"

        elif self.watsonx_llm.model_id in ["meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-405b-instruct", "meta-llama/llama-3-1-70b-instruct"]:
            self.create_propositions_system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + CREATE_PROPOSITIONS_SYSTEM_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>"
            self.create_propositions_user_prompt = CREATE_PROPOSITIONS_USER_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        
        else:
            print("Something went wrong while creation of prompts!!")


        self.parser = CustomListParser(name = "Custom list parser")
        self.fix_parser = OutputFixingParser.from_llm(parser = self.parser, llm = self.watsonx_llm.llm)


        self.proposal_indexing = PromptTemplate(template = self.create_propositions_system_prompt + "\n" + self.create_propositions_user_prompt,
                                                 input_variables = ["source_language", "input"])
        self.runnable = self.proposal_indexing | self.watsonx_llm.llm | self.fix_parser


    def _create_proposition(self, text):
        try:
            if self.verbose:
                runnable_output = self.runnable.invoke(input = {"input": text, "source_language": self.source_language},
                                                       config={"callbacks": [ConsoleCallbackHandler()]})
            else:
                runnable_output = self.runnable.invoke(input = {"input": text, "source_language": self.source_language})
        except Exception as e:
            print(f"An error occurred while getting propositions: {e}")
            return []
        print(f"Extracted stand-alone statements from provided text: {len(runnable_output)}")
        return runnable_output


    def get_propositions(self, nodes: List[str]):
        essay_propositions = []
        for idx, para in tqdm(enumerate(nodes)):
            try:
                propositions = self._create_proposition(para)
                essay_propositions.extend(propositions)
                print(f"Done with paragraph {idx}")
            except Exception as e:
                print(f"An error occurred while processing paragraph {idx}: {e}")
        
        print(f"You have {len(essay_propositions)} propositions")
        return essay_propositions






