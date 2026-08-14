from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import get_user_model
from django import forms

User = get_user_model()


class RegisterForm(UserCreationForm):
    name = forms.CharField(label='Nome', max_length=150, required=True)
    age = forms.IntegerField(
        label='Idade',
        required=False,
        min_value=1,
        max_value=120,
    )
    gender = forms.ChoiceField(
        label='Gênero',
        choices=User.Gender.choices,
        required=True,
    )

    class Meta:
        model = User
        fields = ('email',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['email'].label = 'E-mail'
        self.fields['password1'].label = 'Senha'
        self.fields['password2'].label = 'Confirmar senha'

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip()
        if not email:
            return email
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError('Este e-mail já está em uso.')
        return email

    def save(self, commit=True):
        user = User.objects.create_user(
            email=self.cleaned_data['email'],
            password=self.cleaned_data['password1'],
            first_name=self.cleaned_data['name'].strip(),
            age=self.cleaned_data.get('age'),
            gender=self.cleaned_data['gender'],
        )
        return user


class LoginForm(AuthenticationForm):
    """Login por e-mail (USERNAME_FIELD)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        field = self.fields.get('username')
        if field is not None:
            field.label = 'E-mail'
