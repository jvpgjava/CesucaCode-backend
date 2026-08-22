import io
import threading

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream

SUPPORTED_EXTENSIONS = {"pdf", "docx", "pptx", "txt"}
DOCLING_EXTENSIONS = {"pdf", "docx", "pptx"}


class UnsupportedFileTypeError(Exception):
    pass


def get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


_converter: DocumentConverter | None = None
_converter_lock = threading.Lock()


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        # Lock só protege a construção (chamada uma vez); o pool de
        # processamento tem várias threads e a primeira requisição de cada
        # uma poderia disparar a inicialização em paralelo sem isso.
        with _converter_lock:
            if _converter is None:
                # Materiais didáticos são PDFs de verdade (texto nativo), não
                # escaneados — OCR custa a maior parte do tempo de
                # processamento (~150-240s -> ~40s sem ele num PDF real de
                # teste) sem trazer nada pra esse caso. Se algum dia entrar
                # PDF escaneado de verdade, isso falha com uma mensagem clara
                # ("pode ser uma imagem escaneada sem OCR") em vez de
                # silenciosamente demorar minutos à toa.
                pdf_options = PdfPipelineOptions(do_ocr=False)
                pdf_options.table_structure_options.mode = TableFormerMode.FAST
                _converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
                )
    return _converter


def convert_document(file, filename: str) -> DoclingDocument:
    ext = get_extension(filename)
    if ext not in DOCLING_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Formato .{ext} não suportado.")

    stream = DocumentStream(name=filename, stream=io.BytesIO(file.read()))
    result = _get_converter().convert(stream)
    return result.document


def extract_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")
