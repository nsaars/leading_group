import asyncio
import json
import os
import re
import logging
from pathlib import Path
from pprint import pprint

import pdfplumber

from .book_image_retriever import extract_images
from utils.ai.ai_assistants.ai_helper_functions import AiHelpers

logging.basicConfig(level=logging.INFO)


def prepare_regex(title) -> str:
    """Экранирует специальные символы и заменяет пробелы на \\s* для использования в регулярных выражениях."""
    return str(re.escape(title).replace('\\ ', '\\s*'))

def set_section_text(structure, prev_level, section_text):
    """Устанавливает текст и длину для предыдущего уровня структуры."""
    length = len(section_text)
    if prev_level['level'] == 'subsection':
        chapter = prev_level['chapter']
        section = prev_level['section']
        subsection = prev_level['subsection']
        structure[chapter]['sections'][section]['subsections'][subsection]['text'] = section_text
        structure[chapter]['sections'][section]['subsections'][subsection]['length'] = length
    elif prev_level['level'] == 'section':
        chapter = prev_level['chapter']
        section = prev_level['section']
        structure[chapter]['sections'][section]['text'] = section_text
        structure[chapter]['sections'][section]['length'] = length
    elif prev_level['level'] == 'chapter':
        chapter = prev_level['chapter']
        structure[chapter]['text'] = section_text
        structure[chapter]['length'] = length


def retrieve_book(pdf_path, structure, image_save_folder, image_descriptions_file_path, create_image_descriptions=False):
    # Чтение текста из PDF
    text_list = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[13:]:
            page_text = page.extract_text()
            if page_text:
                text_list.append(page_text)
    text = ''.join(text_list)

    # Извлечение изображений и описаний
    images = extract_images(pdf_path, image_save_folder, False)
    image_descriptions = {}
    ai_helpers = AiHelpers()

    if not create_image_descriptions:
        with open(image_descriptions_file_path, 'r', encoding='utf-8') as json_file:
            image_descriptions = json.loads(json_file.read())

    # Обработка изображений
    for image in images:
        prev_text_regex = prepare_regex(image['prev_text'])
        match = re.search(prev_text_regex, text)
        if match:
            insertion_point = match.end() - 1
            if create_image_descriptions:
                image_descriptions[image['image_name']] =\
                    asyncio.run(ai_helpers.get_image_description(
                        f'{image_save_folder}/{image["image_name"]}',
                        f'...{text[insertion_point - 400:insertion_point]}')).get('image_description').content
            text = (text[:insertion_point] +
                    f"(изображение:{image['image_name']} описание изображения:{image_descriptions[image['image_name']]})"
                    + text[insertion_point:])
        else:
            logging.warning(f"Не найдено совпадение для изображения с текстом: {image['prev_text']}")

    #Сохранение описаний скриншотов
    if create_image_descriptions:
        with open(image_descriptions_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(image_descriptions, json_file, indent=4, ensure_ascii=False)

    # Обработка структуры книги
    prev_match_end = None
    prev_level = {}

    for chapter_num, chapter_dict in structure.items():
        # Регулярное выражение для главы
        chapter_title_regex = prepare_regex(chapter_dict.get('title', ''))
        chapter_regex = rf"(?i)глава\s*{chapter_num}\s*{chapter_title_regex}"
        match = re.search(chapter_regex, text)
        if match:
            chapter_start = match.start()
            chapter_end = match.end()
            structure[chapter_num]['match'] = match
            # Обработка предыдущего уровня
            section_text = text[prev_match_end:chapter_start]
            if prev_match_end is not None and section_text.strip():
                set_section_text(structure, prev_level, section_text)
            prev_match_end = chapter_end
            prev_level = {'level': 'chapter', 'chapter': chapter_num}
        else:
            logging.warning(f"Не найдено совпадение для главы: {chapter_num}")
            continue

        sections = chapter_dict.get('sections', {})
        if not sections:
            continue

        for section_num, section_dict in sections.items():
            # Регулярное выражение для раздела
            section_title_regex = prepare_regex(section_dict.get('title', ''))
            section_regex = rf"(?i){section_num}\s*{section_title_regex}"
            match = re.search(section_regex, text)
            if match:
                section_start = match.start()
                section_end = match.end()
                structure[chapter_num]['sections'][section_num]['match'] = match
                # Обработка предыдущего уровня
                section_text = text[prev_match_end:section_start]
                if prev_match_end is not None and section_text.strip():
                    set_section_text(structure, prev_level, section_text)
                prev_match_end = section_end
                prev_level = {'level': 'section', 'chapter': chapter_num, 'section': section_num}
            else:
                logging.warning(f"Не найдено совпадение для раздела: {section_num}")
                continue

            subsections = section_dict.get('subsections', {})
            if not subsections:
                continue

            for subsection_num, subsection_dict in subsections.items():
                # Регулярное выражение для подраздела
                subsection_title_regex = prepare_regex(subsection_dict.get('title', ''))
                subsection_regex = rf"(?i){subsection_num}\s*{subsection_title_regex}"
                match = re.search(subsection_regex, text)
                if match:
                    subsection_start = match.start()
                    subsection_end = match.end()
                    structure[chapter_num]['sections'][section_num]['subsections'][subsection_num]['match'] = match
                    # Обработка предыдущего уровня
                    section_text = text[prev_match_end:subsection_start]
                    if prev_match_end is not None and section_text.strip():
                        set_section_text(structure, prev_level, section_text)
                    prev_match_end = subsection_end
                    prev_level = {
                        'level': 'subsection',
                        'chapter': chapter_num,
                        'section': section_num,
                        'subsection': subsection_num
                    }
                else:
                    logging.warning(f"Не найдено совпадение для подраздела: {subsection_num}")
                    continue
    # Обработка последнего уровня
    if prev_match_end is not None:
        section_text = text[prev_match_end:]
        set_section_text(structure, prev_level, section_text)

    return structure

if __name__ == '__main__':
    struct = json.loads(open(r"structure.json", 'r').read())
    current_dir = Path(__file__).resolve().parent

    response = retrieve_book(r"../documents/Руководство_Бухгалтерия_для_Узбекистана_ред_3_0 (2).pdf",
                             struct, r"images",
                             image_descriptions_file_path=os.path.join(current_dir, "image_descriptions.json"),
                             )
    pprint(response['12']['sections']['12.2'])