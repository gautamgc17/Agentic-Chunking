from langchain_core.language_models.base import BaseLanguageModel
from langchain_ibm import ChatWatsonx 
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(".env", raise_error_if_not_found=True), verbose=True)


class WatsonxLLM:
    SUPPORTED_MODELS = ["mistralai/mixtral-8x7b-instruct-v01", "mistralai/mistral-large", "meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-405b-instruct", "meta-llama/llama-3-1-8b-instruct"]

    def __init__(self, model_id: str, credentials: dict) -> BaseLanguageModel:
        self.model_id = model_id
        self.credentials = credentials
        
        try:
            if self.model_id in self.SUPPORTED_MODELS:
                self.llm = ChatWatsonx(
                    model_id = self.model_id,
                    url = self.credentials.get("url"),
                    apikey = self.credentials.get("apikey"),
                    project_id = self.credentials.get("project_id"),
                    params = {
                        GenTextParamsMetaNames.DECODING_METHOD: "greedy",
                        GenTextParamsMetaNames.MAX_NEW_TOKENS: 4096,
                        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
                        GenTextParamsMetaNames.REPETITION_PENALTY: 1.05
                    }
                )
                print("Watsonx LLM initialized successfully.")
            else:
                raise ValueError(f"Model '{self.model_id}' is not supported. Supported models are: {', '.join(self.SUPPORTED_MODELS)}")
        except KeyError as e:
            print(f"Missing required credential: {e}")
            self.llm = None
        except Exception as e:
            print(f"An error occurred while initializing Watsonx LLM: {e}")
            self.llm = None