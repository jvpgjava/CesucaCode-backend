import io
import threading

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.io import DocumentStream

SUPPORTED_EXTENSIONS = {"pdf", "docx", "pptx", "md", "txt"}
# md entra no Docling (e não no split simples do txt) pra preservar os
# títulos (#) como `heading` dos chunks, igual aos demais formatos.
DOCLING_EXTENSIONS = {"pdf", "docx", "pptx", "md"}


class UnsupportedFileTypeError(Exception):
    pass


def get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


class OcrUnavailableError(Exception):
    """O PDF não tem texto extraível e o OCR não pôde rodar (engine ausente, modelos
    não baixados, falha na execução)."""


# Dois conversores, criados sob demanda: o padrão (sem OCR) e o com OCR, que só
# existe se algum PDF vier sem texto (escaneado) — OCR custa minutos por documento.
_converters: dict[bool, DocumentConverter] = {}
_converter_lock = threading.Lock()


def _get_converter(ocr: bool = False) -> DocumentConverter:
    if ocr not in _converters:
        # Lock só protege a construção (chamada uma vez); o pool de
        # processamento tem várias threads e a primeira requisição de cada
        # uma poderia disparar a inicialização em paralelo sem isso.
        with _converter_lock:
            if ocr not in _converters:
                # Materiais didáticos costumam ser PDFs de texto nativo; OCR custa a
                # maior parte do tempo de processamento (~150-240s -> ~40s sem ele
                # num PDF real de teste). Por isso o padrão é sem OCR, e o OCR entra
                # como fallback (ver services._extract_chunks) quando a extração
                # vem vazia — o caso de PDFs escaneados, como o Código Disciplinar.
                pdf_options = PdfPipelineOptions(do_ocr=ocr)
                pdf_options.table_structure_options.mode = TableFormerMode.FAST
                _converters[ocr] = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
                )
    return _converters[ocr]


def convert_document(file, filename: str, *, ocr: bool = False) -> DoclingDocument:
    """`file` pode ser um arquivo aberto ou os bytes já lidos (permite reconverter
    o mesmo PDF com OCR sem reabrir o arquivo)."""
    ext = get_extension(filename)
    if ext not in DOCLING_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Formato .{ext} não suportado.")

    data = file if isinstance(file, (bytes, bytearray)) else file.read()
    stream = DocumentStream(name=filename, stream=io.BytesIO(data))
    if not ocr:
        return _get_converter(False).convert(stream).document
    try:
        return _get_converter(True).convert(stream).document
    except Exception as exc:
        raise OcrUnavailableError(
            "O PDF parece ser uma imagem escaneada e o OCR não está disponível ou falhou "
            f"neste servidor ({type(exc).__name__}). Envie uma versão com texto selecionável."
        ) from exc


def extract_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")
