import pytest

from apps.accounts.models import Course


@pytest.mark.django_db
def test_banco_de_teste_tem_cursos_semeados():
    assert Course.objects.filter(code__in=["cc", "ads"]).count() == 2
