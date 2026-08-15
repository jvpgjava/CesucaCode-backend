#!/usr/bin/env python
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Não foi possível importar o Django. Verifique se o ambiente "
            "virtual está ativado e se as dependências foram instaladas "
            "(pip install -r requirements/dev.txt)."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
