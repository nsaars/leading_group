import base64
import json
import os
from http.client import responses
from pathlib import Path
from typing import List, Tuple, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from data import config

parent_dir = Path(__file__).resolve().parent.parent

class AiHelpers:

    def __init__(self, llm: str = None, prompt_templates_file_path: str = None):

        self.llm_title = "gpt-4o-mini"
        self.prompt_templates_file_path: str = os.path.join(parent_dir, "prompts/helpers_prompt_templates.json")


        if llm is not None:
            self.llm_title = llm
        if prompt_templates_file_path is not None:
            self.prompt_templates_file_path = prompt_templates_file_path

        self._llm = ChatOpenAI(model=self.llm_title)

        with open(self.prompt_templates_file_path, "r", encoding='utf-8') as f:
            self._prompt_templates = json.load(f)

    async def get_message_type(self, text: str, history: List[Tuple[str, str]] = None) -> Dict:
        prompt_template = ChatPromptTemplate(
            history + [("system", self._prompt_templates['type_detector']), ("user", "{message}")]
        )

        response = await ({"message": RunnablePassthrough()}
                               | prompt_template
                               | self._llm
                               ).ainvoke(text)

        return {'type': response.content}

    async def get_chat_summary(self, history: List[Tuple[str, str]] = None) -> Dict:
        prompt_template = ChatPromptTemplate([("system", self._prompt_templates['summary_system']),
                                              ("user", "{message}")])
        history_text = ''.join([f'{role}\n{message}\n\n' for role, message in history])

        response = await ({"message": RunnablePassthrough()}
                                  | prompt_template
                                  | self._llm
                                  ).ainvoke(
            f"'{self._prompt_templates['summary']}\nИстория чата с клиентом:\n{history_text}'")

        return {'summary': response.content}

    async def get_proper_question(self, text: str, history: List[Tuple[str, str]] = None) -> Dict:
        prompt_template = ChatPromptTemplate(
            history + [('user', "{message}"), ("system", self._prompt_templates['change_question_system']), ("user", self._prompt_templates['change_question'])])

        response = await (
                {"message": RunnablePassthrough()}
                | prompt_template
                | self._llm
        ).ainvoke(text)

        return {'embedding_text': response.content}

    async def get_image_description(self, image_path: str, prev_text: str) -> Dict:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        messages = [
            SystemMessage(content=self._prompt_templates['image_description']),
            HumanMessage(
                content=[
                    {"type": "text", "text": prev_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ],
            ),
        ]

        # Отправка запроса модели
        response = await self._llm.ainvoke(messages)
        return {'image_description': response.content}



    @staticmethod
    def _format_history(history: List[Tuple[str, str]]) -> str:
        if not history:
            return ''
        return '\n'.join(f"{'Клиент' if role == 'user' else 'Ты'}:\n{message}" for role, message in history)
