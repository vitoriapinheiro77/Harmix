import json

from django import template
from django.utils.html import escape
from django.utils.safestring import mark_safe

register = template.Library()


@register.filter
def debug_format(value):
    """Render pipeline debug values without assuming list/dict shape in templates."""
    if isinstance(value, (dict, list, tuple)):
        formatted = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        return mark_safe(f'<pre class="debug-json">{escape(formatted)}</pre>')
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    if value is None:
        return '—'
    return value
