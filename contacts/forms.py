from django import forms
from .models import *


class ContactForm(forms.ModelForm):
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea(attrs={"rows": 5}))
    sender = forms.EmailField()

    class Meta:
        model = Contact
        fields = ['subject', 'message', 'sender']
