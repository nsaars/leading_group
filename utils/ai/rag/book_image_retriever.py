import fitz


def get_prev_text(sorted_blocks, prev_i, doc, page_num):
    prev_text = ''
    while len(prev_text.strip()) < 20:
        block_text = ''
        if prev_i <= 0:
            sorted_blocks = sorted(doc.load_page(page_num - 1).get_text("dict")['blocks'],
                                   key=lambda item: item['bbox'][3])
            return get_prev_text(sorted_blocks, len(sorted_blocks) - 1, doc, page_num - 1) + prev_text

        lines = sorted_blocks[prev_i].get('lines')
        if not lines:
            prev_i -= 1
            continue
        for line in lines:
            for span in line['spans']:
                span_text = span['text'].strip()
                if span_text:
                    block_text += span_text + ' '
        prev_text = block_text + ' ' + prev_text
        prev_i -= 1
    return prev_text


def extract_images(pdf_path, save_folder, save=True):
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(13, len(doc)):
        page = doc.load_page(page_num)
        page_image_count = 0
        sorted_blocks = sorted(page.get_text("dict")['blocks'], key=lambda item: item['bbox'][3])
        for i in range(len(sorted_blocks)):
            if sorted_blocks[i].get('image'):
                page_image_count += 1
                prev_text = get_prev_text(sorted_blocks, i - 1, doc, page_num)
                image_name = f"image_{page.number + 1}_{page_image_count}.{sorted_blocks[i]['ext']}"
                image_path = f"{save_folder}/{image_name}"
                if save:
                    with open(image_path, "wb") as img_file:
                        img_file.write(sorted_blocks[i]['image'])
                images.append({'image_path': image_path, 'prev_text': prev_text, 'image_name': image_name})
    return images
