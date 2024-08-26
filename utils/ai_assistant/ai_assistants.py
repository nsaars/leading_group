import json
import os
from datetime import datetime
from typing import List, Tuple, Dict
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from utils.ai_assistant.rag.book_retriever import retrieve_book_text

current_dir = os.path.dirname(os.path.abspath(__file__))

class AiQuestionAnswering:
    def __init__(self, llm: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-large",
                 prompt_templates_file_path: str = os.path.join(current_dir, "qa_prompt_templates.json"),
                 search_quantity: int = 3):
        self._store = LocalFileStore(os.path.join(current_dir, "cache"))
        self._underlying_embeddings = OpenAIEmbeddings(model=embedding_model)
        self._cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self._underlying_embeddings, self._store, namespace=self._underlying_embeddings.model
        )
        self._llm = ChatOpenAI(model=llm)

        self._vectorstore = Chroma.from_documents(documents=self._get_docs_from_book_structure(),
                                                  embedding=self._cached_embedder)
        self._retriever = self._vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": search_quantity})

        with open(prompt_templates_file_path, "r", encoding='utf-8') as f:
            self._prompt_templates = json.load(f)

        self._system_prompt = ("system", self._prompt_templates['qa_system'])

    def _get_docs_from_book_structure(self):
        pdf_path = r"C:\Users\Administrator\Desktop\New folder (12)\Руководство_Бухгалтерия_для_Узбекистана_ред_3_0 (" \
                   r"2).pdf "
        structure = json.loads(open(os.path.join(current_dir, 'rag/structure.json'), 'r', encoding='utf-8').read())

        self._book_text = retrieve_book_text(pdf_path, structure, os.path.join(current_dir, 'rag/images'))
        self._docs = []
        for chapter, chapter_dict in structure.items():
            chapter_text = chapter_dict.get('text')
            if chapter_text:
                text_beginning = f"(Следующий текст с главы {chapter} '{chapter_dict['title']}'.)"
                self._docs.append(Document(page_content=text_beginning + chapter_text,
                                           metadata={'id': chapter, 'chapter': chapter, }))
                continue
            sections = chapter_dict.get('sections')
            for section, section_dict in sections.items():
                section_text = section_dict.get('text')
                if section_text:
                    text_beginning = f"(Следующий текст с раздела {section} '{section_dict['title']}'.\n" \
                                     f"Глава этого раздела: {chapter} '{chapter_dict['title']}')"
                    self._docs.append(Document(page_content=text_beginning + section_text,
                                               metadata={'id': section, 'chapter': chapter, 'section': section}))
                    continue
                subsections = section_dict.get('subsections')
                for subsection, subsection_dict in subsections.items():
                    subsection_text = subsection_dict.get('text')
                    text_beginning = f"(Следующий текст с подраздела {subsection} '{subsection_dict['title']}'.\n" \
                                     f"Раздел этого подраздела: {section} '{section_dict['title']}'.\n" \
                                     f"Глава этого раздела: {chapter} '{chapter_dict['title']}')"
                    self._docs.append(Document(page_content=text_beginning + subsection_text,
                                               metadata={'id': section, 'chapter': chapter, 'section': section}))
        return self._docs

    @staticmethod
    def format_docs(similar_docs: List[Document]) -> str:
        return "\n\n".join(f"{doc.page_content}" for doc in similar_docs)

    @staticmethod
    def get_formatted_datetime():
        now = datetime.now()
        weekday = now.weekday()
        days = {
            0: "понедельник", 1: "вторник", 2: "среда", 3: "четверг",
            4: "пятница", 5: "суббота", 6: "воскресенье"
        }
        formatted_date = now.strftime("%Y-%m-%d")
        formatted_time = now.strftime("%H:%M")
        if now.hour > 18:
            formatted_time = "уже" + formatted_time
        return days[weekday] + ", " + formatted_date, formatted_time

    async def get_default_response(self, text: str, history: List[Tuple[str, str]] = None) -> Dict:
        if history is None:
            history = []

        default_template = ChatPromptTemplate(
            [self._system_prompt]
            + history + [('user', "{message}")])

        return {'default_response': await (
                {"message": RunnablePassthrough()}
                | default_template
                | self._llm
        ).ainvoke(text)}

    async def get_question_response(self, text: str, history: List[Tuple[str, str]] = None) -> Dict:
        if history is None:
            history = []

        question_template = ChatPromptTemplate(
            [self._system_prompt] +
            history + [('user', self._prompt_templates['qa_question'] +
                        """База знаний для ответа на вопрос:\n{context}\n\nВопрос клиента: {message}""")])
        response = await (
                {"context": self._retriever | self.format_docs, "message": RunnablePassthrough()}
                | question_template
                | self._llm
        ).ainvoke(text)
        return {'question_response': response}


class AiHelpers:
    def __init__(self, llm: str = "gpt-4o-mini",
                 prompt_templates_file_path: str = os.path.join(current_dir, "helpers_prompt_templates.json")):
        self._llm = ChatOpenAI(model=llm)

        with open(prompt_templates_file_path, "r", encoding='utf-8') as f:
            self._prompt_templates = json.load(f)

    async def get_message_type(self, text: str, history: List[Tuple[str, str]] = None) -> Dict:
        prompt_template = ChatPromptTemplate(
            history + [("system", self._prompt_templates['type_detector']), ("user", "{message}")]
        )
        return {'type': await ({"message": RunnablePassthrough()}
                               | prompt_template
                               | self._llm
                               ).ainvoke(text)}

    async def get_chat_summary(self, history: List[Tuple[str, str]] = None) -> Dict:
        prompt_template = ChatPromptTemplate([("system", self._prompt_templates['summary_system']),
                                              ("user", "{message}")])
        history_text = ''.join([f'{role}\n{message}\n\n' for role, message in history])
        return {'summary': await ({"message": RunnablePassthrough()}
                                  | prompt_template
                                  | self._llm
                                  ).ainvoke(
            f"'{self._prompt_templates['summary']}\nИстория чата с клиентом:\n{history_text}'")}

    @staticmethod
    def _format_history(history: List[Tuple[str, str]]) -> str:
        if not history:
            return ''
        return '\n'.join(f"{'Клиент' if role == 'user' else 'Ты'}:\n{message}" for role, message in history)
