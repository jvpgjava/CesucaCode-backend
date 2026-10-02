from django.contrib.postgres.search import SearchVector
from django.db import migrations


def popular_search_vector(apps, schema_editor):
    """Preenche o tsvector dos chunks que já existem (os novos são preenchidos na
    ingestão). Não depende de embedding, então não exige reprocessar nada."""
    DocumentChunk = apps.get_model("documents", "DocumentChunk")
    DocumentChunk.objects.update(
        search_vector=SearchVector("heading", weight="A", config="portuguese")
        + SearchVector("content", weight="B", config="portuguese")
    )


class Migration(migrations.Migration):
    dependencies = [
        ("documents", "0005_chunk_indices_busca_hibrida"),
    ]

    operations = [
        migrations.RunPython(popular_search_vector, migrations.RunPython.noop),
    ]
