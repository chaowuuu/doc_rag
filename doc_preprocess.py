from pathlib import Path
from typing import List


class DocumentProcess:
    def __init__(self):
        pass

    def load_documents(self, directory_path: str) -> List[str]:
        documents = []
        for file_path in Path(directory_path).rglob('*.md'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
            except Exception as e:
                print(e)

        return documents

processor = DocumentProcess()
print(len(processor.load_documents(r"")))
