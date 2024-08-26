import json
import re
from pprint import pprint

import pdfplumber

from .book_image_retriever import extract_images


def retrieve_book_text(pdf_path, structure, image_save_folder):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages[13:]:
            text += page.extract_text()

    images = extract_images(pdf_path, image_save_folder)

    for image in images:
        prev_text_regex = image['prev_text'].replace(' ', '\\s*').replace(' ', '\\s*').replace('(', '\\(').replace(')', '\\)')
        match = re.search(prev_text_regex, text)
        text = text[:match.end() - 1] + f"(изображение:{image['image_name']})" + text[match.end():]
    temp = None

    for chapter, chapter_dict in structure.items():
        chapter_title_regex = chapter_dict.get('title').replace(' ', '\\s*')
        chapter_regex = rf"(?i)глава\s*{chapter}\s*{chapter_title_regex}"
        match = re.search(chapter_regex, text)
        structure[chapter]['match'] = match
        if temp is not None:
            if temp.get('subsection'):
                structure[temp['chapter']]['sections'][temp['section']]['subsections'][temp['subsection']]['text'] \
                    = text[temp['end']:match.start()]
                structure[temp['chapter']]['sections'][temp['section']]['subsections'][temp['subsection']]['length'] \
                    = match.start() - temp['end']
            elif temp.get('section'):
                structure[temp['chapter']]['sections'][temp['section']]['text'] = text[temp['end']:match.start()]
                structure[temp['chapter']]['sections'][temp['section']]['length'] = match.start() - temp['end']
            else:
                structure[temp['chapter']]['text'] = text[temp['end']:match.start()]
                structure[temp['chapter']]['length'] = match.start() - temp['end']
            temp = None

        sections = chapter_dict.get('sections')
        if not sections:
            temp = {'chapter': chapter, 'end': match.end()}
            continue
        for section, section_dict in sections.items():
            section_title_regex = section_dict.get('title') \
                .replace(' ', '\\s*').replace('(', '\\(').replace(')', '\\)')
            section_regex = rf"(?i){section}\s*{section_title_regex}"
            match = re.search(section_regex, text)
            structure[chapter]['sections'][section]['match'] = match

            if temp is not None:
                if temp.get('subsection'):
                    structure[temp['chapter']]['sections'][temp['section']]['subsections'][temp['subsection']]['text'] \
                        = text[temp['end']:match.start()]
                    structure[temp['chapter']]['sections'][temp['section']]['subsections'][temp['subsection']]['length'] \
                        = match.start() - temp['end']
                else:
                    structure[temp['chapter']]['sections'][temp['section']]['text'] = text[temp['end']:match.start()]
                    structure[temp['chapter']]['sections'][temp['section']]['length'] = match.start() - temp['end']
                temp = None

            subsections: dict = section_dict.get('subsections')
            if not subsections:
                temp = {'chapter': chapter, 'section': section, 'end': match.end()}
                continue
            for subsection, subsection_dict in subsections.items():
                subsection_title_regex = subsection_dict.get('title') \
                    .replace(' ', '\\s*').replace('(', '\\(').replace(')', '\\)')
                subsection_regex = rf"(?i){subsection}\s*{subsection_title_regex}"
                match = re.search(subsection_regex, text)
                structure[chapter]['sections'][section]['subsections'][subsection]['match'] = match

                if temp is not None:
                    try:
                        structure[temp['chapter']]['sections'][temp['section']]['subsections'][temp['subsection']]['text'] \
                            = text[temp['end']:match.start()]
                    except:
                        print(match, subsection_regex)
                    structure[temp['chapter']]['sections'][temp['section']]['subsections'][temp['subsection']]['length'] \
                        = match.start() - temp['end']

                temp = {'chapter': chapter, 'section': section, 'subsection': subsection, 'end': match.end()}
    if temp is not None:
        if temp.get('subsection'):
            structure[temp['chapter']]['sections'][temp['section']]['subsections'][temp['subsection']]['text'] \
                = text[temp['end']:match.start()]
            structure[temp['chapter']]['sections'][temp['section']]['subsections'][temp['subsection']]['length'] \
                = match.start() - temp['end']
        elif temp.get('section'):
            structure[temp['chapter']]['sections'][temp['section']]['text'] = text[temp['end']:match.start()]
            structure[temp['chapter']]['sections'][temp['section']]['length'] = match.start() - temp['end']
        else:
            structure[temp['chapter']]['text'] = text[temp['end']:match.start()]
            structure[temp['chapter']]['length'] = match.start() - temp['end']

    return structure

