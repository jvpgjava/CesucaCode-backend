# Materiais de seed

Cada `.md` daqui é o que o `seed_materials` ingere (rápido, com títulos preservados).
Em `originals/` ficam os PDFs de origem, com o mesmo nome-base, na íntegra. Eles **não**
são ingeridos: servem de referência e para regerar o `.md` se preciso.

| Markdown | PDF original | Observação |
|---|---|---|
| `avaliacao.md` | `originals/avaliacao.pdf` | exportação do Docling |
| `manual.md` | `originals/manual.pdf` | exportação do Docling |
| `horario-cc-2026-2.md` | `originals/horario-cc-2026-2.pdf` | exportação do Docling; a tabela perde a divisão por semestre (o PDF a mantém) |
| `plano-de-ensino-cc.md` | `originals/plano-de-ensino-cc.pdf` | exportação do Docling |
| `codigo-disciplinar-interno.md` | `originals/codigo-disciplinar-interno.pdf` | PDF **escaneado**: gerado uma única vez com Docling + OCR (`do_ocr=True`, ~4 min), marcadores `<!-- image -->` removidos. Ruído de OCR na página de assinaturas |

O ingestor roda com OCR desligado, por isso PDFs escaneados precisam virar `.md` antes.
