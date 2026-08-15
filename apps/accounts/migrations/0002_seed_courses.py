from django.db import migrations

COURSES = [
    ("cc", "Ciência da Computação"),
    ("ads", "Análise e Desenvolvimento de Sistemas"),
]


def seed_courses(apps, schema_editor):
    Course = apps.get_model("accounts", "Course")
    for code, name in COURSES:
        Course.objects.get_or_create(code=code, defaults={"name": name})


def remove_courses(apps, schema_editor):
    Course = apps.get_model("accounts", "Course")
    Course.objects.filter(code__in=[code for code, _ in COURSES]).delete()


class Migration(migrations.Migration):
    dependencies = [("accounts", "0001_initial")]
    operations = [migrations.RunPython(seed_courses, remove_courses)]
