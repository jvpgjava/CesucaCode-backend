from django.db import migrations, models


def copy_course_to_courses(apps, schema_editor):
    Document = apps.get_model("documents", "Document")
    for document in Document.objects.all():
        document.courses.add(document.course_id)


class Migration(migrations.Migration):
    """Um material passa a valer pra vários cursos (`course` FK -> `courses` M2M).

    Os materiais existentes mantêm o curso que já tinham. Só vai pra frente:
    voltar exigiria decidir qual curso ficaria num material com vários.
    """

    dependencies = [
        ("accounts", "0003_user_avatar"),
        ("documents", "0003_documentchunk_heading"),
    ]

    operations = [
        migrations.AddField(
            model_name="document",
            name="courses",
            field=models.ManyToManyField(
                help_text="Cursos para os quais o material vale (um ou mais).",
                related_name="documents_m2m",
                to="accounts.course",
            ),
        ),
        migrations.RunPython(copy_course_to_courses, migrations.RunPython.noop),
        migrations.RemoveField(model_name="document", name="course"),
        migrations.AlterField(
            model_name="document",
            name="courses",
            field=models.ManyToManyField(
                help_text="Cursos para os quais o material vale (um ou mais).",
                related_name="documents",
                to="accounts.course",
            ),
        ),
    ]
