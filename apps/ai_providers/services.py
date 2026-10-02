from django.conf import settings

from .exceptions import ProviderConfigurationError


def get_chat_model():
    provider = settings.LLM_PROVIDER

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=settings.LLM_MODEL, google_api_key=settings.GOOGLE_API_KEY)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.LLM_MODEL, openai_api_key=settings.OPENAI_API_KEY)

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.LLM_MODEL, anthropic_api_key=settings.ANTHROPIC_API_KEY)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL)

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.DEEPSEEK_API_KEY,
            openai_api_base="https://api.deepseek.com/v1",
        )

    if provider == "abacusai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.ABACUSAI_API_KEY,
            openai_api_base="https://routellm.abacus.ai/v1",
        )

    raise ProviderConfigurationError(f"Provider de chat '{provider}' não suportado.")


def get_embedding_model():
    provider = settings.EMBEDDING_PROVIDER

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)

    raise ProviderConfigurationError(f"Provider de embedding '{provider}' não suportado.")


def validate_embedding_dimensions(vector) -> None:
    """Falha com mensagem clara se o modelo devolver um vetor de tamanho diferente
    do da coluna `embedding` (EMBEDDING_DIMENSIONS). Sem isso, o erro aparece só
    como falha obscura de SQL ao gravar/buscar."""
    expected = settings.EMBEDDING_DIMENSIONS
    if len(vector) != expected:
        raise ProviderConfigurationError(
            f"O modelo de embedding '{settings.EMBEDDING_MODEL}' devolveu vetores de "
            f"{len(vector)} dimensões, mas EMBEDDING_DIMENSIONS={expected}. Ajuste o "
            "modelo ou a dimensão (trocar a dimensão exige nova migration e reprocessar os materiais)."
        )
