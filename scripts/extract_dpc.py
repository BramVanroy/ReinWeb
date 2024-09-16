from pathlib import Path

from datasets import Dataset
from lxml import etree as ET


def extract_text(xml_f: str):
    """Read XML and extract all segments and their text.
    The intention was to separate paragraphs by newlines but in reality it seems that this metadata has not been
    preserved since all texts are only one paragraph long according to the XML structure, even though that is not
    the case in the actual text - but sadly that cannot be inferred from the XML. (I.e. the XML is not a reliable
    source of information about the text structure - only one `p` tag per document.)
    """
    xml = ET.parse(xml_f)
    root = xml.getroot()
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    text = []
    num_sentences = 0
    num_paragraphs = 0
    for p in root.findall(".//tei:p", ns):
        para = []
        for sent in p.findall(".//tei:seg[@type='original']", ns):
            sent_text = sent.text.strip()
            if sent_text and not sent_text.startswith("<!"):
                para.append(sent_text)
                num_sentences += 1
        text.append(" ".join(para))
        num_paragraphs += 1

    text = "\n".join(text)

    return {"text": text, "text_len": len(text), "num_sentences": num_sentences, "num_paragraphs": num_paragraphs}


def main(dpc_core_dir: str):
    pdin = Path(dpc_core_dir)

    data = []
    for subcorpus in pdin.iterdir():
        if subcorpus.is_dir():
            data.extend(
                [
                    {"path": str(xml_pf.resolve()), "subcorpus": subcorpus.stem}
                    for xml_pf in subcorpus.glob("*-nl-tei.xml")
                ]
            )

    ds = Dataset.from_list(data)
    ds = ds.map(extract_text, input_columns="path", num_proc=36)
    ds.push_to_hub("BramVanroy/DPC1.0-dutch", private=True)


if __name__ == "__main__":
    main("/home/nobackup/corpora/DPC1.0/data/core")
