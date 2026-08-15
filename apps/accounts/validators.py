from django.core.validators import RegexValidator

rgm_validator = RegexValidator(
    regex=r"^\d+$",
    message="O RGM deve conter apenas números.",
)
