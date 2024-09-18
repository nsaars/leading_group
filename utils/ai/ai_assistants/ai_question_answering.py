import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from utils.ai.ai_assistants.ai_helper_functions import AiHelpers
from utils.ai.rag.book_retriever import retrieve_book

from data import config

parent_dir = Path(__file__).resolve().parent.parent

class AiQuestionAnswering:
    def __init__(self, llm: str = "gpt-4o",
                 embedding_model: str = "text-embedding-3-large",
                 prompt_templates_file_path: str = os.path.join(parent_dir, "prompts/qa_prompt_templates.json"),
                 search_quantity: int = 2):
        self._store = LocalFileStore(os.path.join(parent_dir, "cache"))
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
        pdf_path = os.path.join(parent_dir, 'documents/Руководство_Бухгалтерия_для_Узбекистана_ред_3_0 (2).pdf')
        structure = json.loads(open(os.path.join(parent_dir, 'rag/structure.json'), 'r', encoding='utf-8').read())

        self._book_text = retrieve_book(pdf_path, structure, os.path.join(parent_dir, 'rag/images'),
                                        os.path.join(parent_dir, "rag/image_descriptions.json"))
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
                        """\nБаза знаний для ответа на вопрос:\n{context}\n\nВопрос клиента: {message}""")])
        embedding_text = (await AiHelpers().get_embedding_text(text, history)).get('embedding_text').content
        context_docs = await self._retriever.ainvoke(embedding_text)
        formatted_context = RunnableLambda(lambda x: self.format_docs(context_docs))
        response = await (
                {"context": formatted_context, "message": RunnablePassthrough()}
                | question_template
                | self._llm
        ).ainvoke(text)
        return {'question_response': response}