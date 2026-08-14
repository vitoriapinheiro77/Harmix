from django.contrib.auth.models import AbstractUser
from django.db import models

from .managers import UserManager


class User(AbstractUser):
    class Gender(models.TextChoices):
        FEMALE = 'female', 'Feminino'
        MALE = 'male', 'Masculino'
        NON_BINARY = 'non_binary', 'Não-binário'
        PREFER_NOT = 'prefer_not', 'Prefiro não informar'
        OTHER = 'other', 'Outro'

    email = models.EmailField(unique=True)
    age = models.PositiveSmallIntegerField(null=True, blank=True)
    gender = models.CharField(
        max_length=20,
        choices=Gender.choices,
        default=Gender.PREFER_NOT,
    )

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS: tuple[str, ...] = ()

    def __str__(self) -> str:
        name = (self.first_name or '').strip()
        return name or self.email
