import fitz
import re

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()

        if page_text.strip():
            text += f"\n\n--- Page {page_num + 1} ---\n"
            text += page_text

    doc.close()
    return text


def clean_text(text):
    text = re.sub(r'--- Page \d+ ---', '', text)
    text = re.sub(r'\bPage \d+\b', '', text)
    text = re.sub(r'ITTF[\s\w]*2025', '', text)

    # Keep double newlines for section detection
    text = re.sub(r'\n{3,}', '\n\n', text)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' \n', '\n', text)

    return text.strip()


def remove_front_matter(text):
    # Skip everything before the first real section "0." or "1."
    # Looks for pattern like "0.\n" or "1.\n" or "1. CONSTITUTION" etc.
    match = re.search(r'\n0\.[\s\n]|\n1\.[\s\n]', text)
    if match:
        return text[match.start():].strip()
    return text


def remove_table_of_contents(text):
    # TOC lines look like "Some Title ... 123" or "Some Title\n123"
    # Remove lines that are just a title + page number
    text = re.sub(r'^.{5,60}\.{2,}\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # lone page numbers
    return text


def chunk_by_section(text, max_chunk_size=2000):
    # Split on numbered headings like "1.", "2.3", "4.1.2"
    section_pattern = r'(?=\n\d+(\.\d+)*[\.\s][A-Z])'
    sections = re.split(section_pattern, text)

    chunks = []
    current_chunk = ""

    for section in sections:
        if not section or not section.strip():
            continue

        if len(current_chunk) + len(section) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk += "\n" + section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ===== MAIN =====
pdf_path = "ITTF_Rulebook_2025.pdf"

raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_text(raw_text)
cleaned_text = remove_front_matter(cleaned_text)   # drop cover page + pub history
cleaned_text = remove_table_of_contents(cleaned_text)  # drop TOC lines
chunks = chunk_by_section(cleaned_text, max_chunk_size=2000)

print(f"Total characters extracted: {len(cleaned_text)}")
print(f"Total chunks created: {len(chunks)}")
print(f"Avg chunk size: {len(cleaned_text) // len(chunks)} chars")

print("\nFirst chunk preview:\n")
print(chunks[0])

print("\nSecond chunk preview:\n")
print(chunks[1])