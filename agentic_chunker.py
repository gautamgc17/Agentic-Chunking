import os
import uuid
from tqdm import tqdm
from typing import Optional, List
from langchain_ibm import ChatWatsonx 
from langchain.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain.callbacks.tracers import ConsoleCallbackHandler
from models import *
from prompts import *


class AgenticChunker:
    def __init__(self, llm: ChatWatsonx, source_language: str = "English", verbose: bool = True):
        
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True           # Whether or not to update/refine summaries and titles as you get new information

        self.watsonx_llm = llm
        self.source_language = source_language
        self.verbose = verbose


    def _add_propositions(self, propositions):
        for proposition in propositions:
            self._add_proposition(proposition.strip())
    

    def _add_proposition(self, proposition):
        if self.verbose:
            print (f"\nAdding: '{proposition}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.verbose:
                print ("No chunks, creating a new one..")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        # If a chunk was found then add the proposition to it
        if chunk_id:
            if self.verbose:
                print (f"Chunk Found ({self.chunks[chunk_id]['chunk_id'].strip()}), adding to: {self.chunks[chunk_id]['title'].strip()}")
            self._add_proposition_to_chunk(chunk_id, proposition.strip())
            return
        else:
            if self.verbose:
                print ("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition.strip())
        

    def _add_proposition_to_chunk(self, chunk_id, proposition):
        # Add then
        self.chunks[chunk_id]['propositions'].append(proposition)

        # Then grab a new summary
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])


    def _update_chunk_summary(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the summary or else they could get stale
        """
        if self.watsonx_llm.model_id in ["mistralai/mixtral-8x7b-instruct-v01", "mistralai/mistral-large"]:
            self.update_chunk_summary_system_prompt = "[INST]<<SYS>>" + UPDATE_CHUNK_SUMMARY_SYSTEM_PROMPT
            self.update_chunk_summary_user_prompt = UPDATE_CHUNK_SUMMARY_USER_PROMPT + "[/INST]"

        elif self.watsonx_llm.model_id in ["meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-405b-instruct", "meta-llama/llama-3-1-70b-instruct"]:
            self.update_chunk_summary_system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + UPDATE_CHUNK_SUMMARY_SYSTEM_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>"
            self.update_chunk_summary_user_prompt = UPDATE_CHUNK_SUMMARY_USER_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        
        else:
            print("Something went wrong while creation of prompts!!")

        self.update_chunk_summary_prompt = PromptTemplate(template = self.update_chunk_summary_system_prompt + "\n" + self.update_chunk_summary_user_prompt,
                                                  input_variables = ["proposition", "current_summary", "source_language"])

        self.runnable = self.update_chunk_summary_prompt | self.watsonx_llm.llm

        if self.verbose:
            new_chunk_summary = self.runnable.invoke(input = {"proposition": "\n".join(chunk['propositions']), "current_summary": chunk['summary'], "source_language": self.source_language},
                                                    config={"callbacks": [ConsoleCallbackHandler()]}).content
        else:
            new_chunk_summary = self.runnable.invoke(input = {"proposition": "\n".join(chunk['propositions']), "current_summary": chunk['summary'], "source_language": self.source_language}).content 
        return new_chunk_summary.strip()
    

    def _update_chunk_title(self, chunk):
        """
        If you add a new proposition to a chunk, you may want to update the title or else it can get stale
        """
        if self.watsonx_llm.model_id in ["mistralai/mixtral-8x7b-instruct-v01", "mistralai/mistral-large"]:
            self.update_chunk_title_system_prompt = "[INST]<<SYS>>" + UPDATE_CHUNK_TITLE_SYSTEM_PROMPT
            self.update_chunk_title_user_prompt = UPDATE_CHUNK_TITLE_USER_PROMPT + "[/INST]"

        elif self.watsonx_llm.model_id in ["meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-405b-instruct", "meta-llama/llama-3-1-70b-instruct"]:
            self.update_chunk_title_system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + UPDATE_CHUNK_TITLE_SYSTEM_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>"
            self.update_chunk_title_user_prompt = UPDATE_CHUNK_TITLE_USER_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        
        else:
            print("Something went wrong while creation of prompts!!")

        self.update_chunk_title_prompt = PromptTemplate(template = self.update_chunk_title_system_prompt + "\n" + self.update_chunk_title_user_prompt,
                                                        input_variables = ["proposition", "current_summary", "current_title", "source_language"])

        self.runnable = self.update_chunk_title_prompt | self.watsonx_llm.llm

        if self.verbose:
            updated_chunk_title = self.runnable.invoke(input = {"proposition": "\n".join(chunk['propositions']), "current_summary" : chunk['summary'], "current_title" : chunk['title'], "source_language": self.source_language},
                                                       config = {"callbacks": [ConsoleCallbackHandler()]}).content
        else:
            updated_chunk_title = self.runnable.invoke(input = {"proposition": "\n".join(chunk['propositions']), "current_summary" : chunk['summary'], "current_title" : chunk['title'], "source_language": self.source_language}).content

        return updated_chunk_title.strip()


    def _get_new_chunk_summary(self, proposition):
        if self.watsonx_llm.model_id in ["mistralai/mixtral-8x7b-instruct-v01", "mistralai/mistral-large"]:
            self.new_chunk_summary_system_prompt = "[INST]<<SYS>>" + NEW_CHUNK_SUMMARY_SYSTEM_PROMPT
            self.new_chunk_summary_user_prompt = NEW_CHUNK_SUMMARY_USER_PROMPT + "[/INST]"

        elif self.watsonx_llm.model_id in ["meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-405b-instruct", "meta-llama/llama-3-1-70b-instruct"]:
            self.new_chunk_summary_system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + NEW_CHUNK_SUMMARY_SYSTEM_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>"
            self.new_chunk_summary_user_prompt = NEW_CHUNK_SUMMARY_USER_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        
        else:
            print("Something went wrong while creation of prompts!!")

        self.get_new_chunk_summary_prompt = PromptTemplate(template = self.new_chunk_summary_system_prompt + "\n" + self.new_chunk_summary_user_prompt,
                                                          input_variables = ["proposition", "source_language"])

        self.runnable = self.get_new_chunk_summary_prompt | self.watsonx_llm.llm

        if self.verbose:
            new_chunk_summary = self.runnable.invoke(input = {"proposition": proposition, "source_language": self.source_language}, 
                                            config = {"callbacks": [ConsoleCallbackHandler()]}).content
        else:
            new_chunk_summary = self.runnable.invoke(input = {"proposition": proposition, "source_language": self.source_language}).content


        return new_chunk_summary.strip()
    

    def _get_new_chunk_title(self, summary):
        if self.watsonx_llm.model_id in ["mistralai/mixtral-8x7b-instruct-v01", "mistralai/mistral-large"]:
            self.new_chunk_title_system_prompt = "[INST]<<SYS>>" + NEW_CHUNK_TITLE_SYSTEM_PROMPT
            self.new_chunk_title_user_prompt = NEW_CHUNK_TITLE_USER_PROMPT + "[/INST]"

        elif self.watsonx_llm.model_id in ["meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-405b-instruct", "meta-llama/llama-3-1-70b-instruct"]:
            self.new_chunk_title_system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + NEW_CHUNK_TITLE_SYSTEM_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>"
            self.new_chunk_title_user_prompt = NEW_CHUNK_TITLE_USER_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        else:
            print("Something went wrong while creation of prompts!!")

        self.get_new_chunk_title_prompt = PromptTemplate(template = self.new_chunk_title_system_prompt + "\n" + self.new_chunk_title_user_prompt,
                                                         input_variables = ["summary", "source_language"])

        self.runnable = self.get_new_chunk_title_prompt | self.watsonx_llm.llm

        if self.verbose:
            new_chunk_title = self.runnable.invoke(input = {"summary": summary, "source_language": self.source_language},
                                                   config = {"callbacks": [ConsoleCallbackHandler()]}).content
        else:
            new_chunk_title = self.runnable.invoke(input = {"summary": summary, "source_language": self.source_language}).content

        return new_chunk_title.strip()


    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit] 
        new_chunk_summary = self._get_new_chunk_summary(proposition.strip())
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary.strip())

        self.chunks[new_chunk_id] = {
            'chunk_id' : new_chunk_id,
            'propositions': [proposition.strip()],
            'title' : new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index' : len(self.chunks)
        }
        if self.verbose:
            print (f"Created new chunk ({new_chunk_id}): {new_chunk_title}")
    

    def _get_chunk_outline(self):
        """
        Get a string which represents the chunks you currently have.
        This will be empty when you first start off
        """
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ID: {chunk['chunk_id'].strip()}\nChunk Name: {chunk['title'].strip()}\nChunk Summary: {chunk['summary'].strip()}\n\n"""
        
            chunk_outline += single_chunk_string
        
        return chunk_outline.strip()


    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self._get_chunk_outline()

        if self.watsonx_llm.model_id in ["mistralai/mixtral-8x7b-instruct-v01", "mistralai/mistral-large"]:
            self.find_relevant_chunk_system_prompt = "[INST]<<SYS>>" + FIND_RELEVANT_CHUNK_SYSTEM_PROMPT
            self.find_relevant_chunk_user_prompt = FIND_RELEVANT_CHUNK_USER_PROMPT + "[/INST]"

        elif self.watsonx_llm.model_id in ["meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-405b-instruct", "meta-llama/llama-3-1-70b-instruct"]:
            self.find_relevant_chunk_system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + FIND_RELEVANT_CHUNK_SYSTEM_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>"
            self.find_relevant_chunk_user_prompt = FIND_RELEVANT_CHUNK_USER_PROMPT.strip() + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        else:
            print("Something went wrong while creation of prompts!!")

        self.find_relevant_chunk_prompt = PromptTemplate(template = self.find_relevant_chunk_system_prompt + "\n" + self.find_relevant_chunk_user_prompt, 
                                                         input = ["proposition", "current_chunk_outline", "source_language"])

        self.runnable = self.find_relevant_chunk_prompt | self.watsonx_llm.llm

        if self.verbose:
            chunk_found = self.runnable.invoke(input = {"proposition": proposition, "current_chunk_outline": current_chunk_outline, "source_language": self.source_language},
                                           config = {"callbacks": [ConsoleCallbackHandler()]}).content
        else:
            chunk_found = self.runnable.invoke(input = {"proposition": proposition, "current_chunk_outline": current_chunk_outline, "source_language": self.source_language}).content

        print("Chunk Found --> ", chunk_found.strip())
        if len(chunk_found.strip()) != self.id_truncate_limit:
            return None

        return chunk_found.strip()
    

    def get_chunks(self, get_type='dict'):
        """
        This function returns the chunks in the format specified by the 'get_type' parameter.
        If 'get_type' is 'dict', it returns the chunks as a dictionary.
        If 'get_type' is 'list_of_strings', it returns the chunks as a list of strings, where each string is a proposition in the chunk.
        """
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x.strip() for x in chunk['propositions']]))
            return chunks
    

    def _pretty_print_chunks(self):
        print (f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions:")
            for prop in chunk['propositions']:
                print(f"    -{prop}")
            print("\n\n")


    def _pretty_print_chunk_outline(self):
        print ("Chunk Outline\n")
        print(self._get_chunk_outline())


    def process_propositions_with_intermediate_steps(self, propositions, get_type='list_of_strings'):
        print("--"*50)
        print("add_propositions")
        print("--"*50)
        self._add_propositions(propositions)

        print("--"*50)
        print("pretty_print_chunks")
        print("--"*50)
        self._pretty_print_chunks()

        print("--"*50)
        print("pretty_print_chunk_outline")
        print("--"*50)
        self._pretty_print_chunk_outline()

        print("--"*50)
        print("get_chunks") 
        print("--"*50)
        print(len(self.get_chunks(get_type='list_of_strings')))
        return self.get_chunks(get_type='list_of_strings')


    def process_propositions(self, propositions, get_type='list_of_strings'):
        print("--"*50)
        print("add_propositions")
        print("--"*50)
        self._add_propositions(propositions)

        print("--"*50)
        print("get_chunks") 
        print("--"*50)
        return self.get_chunks(get_type='list_of_strings')

