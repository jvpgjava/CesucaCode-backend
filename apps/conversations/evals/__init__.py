"""Avaliação (evals) da S.O.F.I.A.: golden set + runner + métricas.

Uso: `manage.py run_evals` (ver README). O pacote consome o chat somente pela
interface pública `services.send_message` e lê o `MessageTrace` da resposta, então
serve para comparar variantes do pipeline (v0/v1/v2) sem depender da sua implementação.
"""
