"""The HTML version of WarriorIQ's emails (owner, 2026-11-15: "make them look
professional, with our logo").

Every message keeps its plain-text body (core/notifications.py sends both);
this builds the HTML alternative from that same text, so the two can never say
different things:

- the WarriorIQ logo and name in a dark header, the site's blue as the accent;
- each paragraph of the text as a paragraph;
- a link on a line of its own as a button, labelled for what it does, with the
  address underneath for mail apps that block buttons;
- a footer with the site's address.

Table layout and inline styles only: that is what Gmail, Outlook and Apple
Mail all render. Every piece of text is escaped.
"""

from __future__ import annotations

import html
import re

_URL = re.compile(r"https?://[^\s<>\"']+")
BLUE = "#2f7cf6"
INK = "#101828"
MUTED = "#5b6577"

# What a button says, by the address it opens.
_BUTTON_LABELS = (
    ("/verify-email/", "Verify your email"),
    ("/reset-password/", "Reset your password"),
    ("/settings/email/confirm/", "Confirm your new email"),
    ("/guardian/approve/", "Review the account"),
    ("/coaching/invite/", "See the invitation"),
    ("/gym/invite/", "See the invitation"),
    ("/progress/", "Follow the analysis"),
    ("/result/", "Open the report"),
)


def _button_label(url: str) -> str:
    return next((label for marker, label in _BUTTON_LABELS if marker in url), "Open WarriorIQ")


def _linkify(text: str) -> str:
    """Escape a line and turn any address in it into a link."""
    out, last = [], 0
    for match in _URL.finditer(text):
        url = match.group(0).rstrip(".,;:)")
        out.append(html.escape(text[last:match.start()]))
        out.append(f'<a href="{html.escape(url, quote=True)}" style="color:{BLUE};text-decoration:underline">'
                   f'{html.escape(url)}</a>')
        last = match.start() + len(url)
    out.append(html.escape(text[last:]))
    return "".join(out)


_LABELLED_URL = re.compile(r"^(?P<label>[^\n:]{2,60}):\s*(?P<url>https?://[^\s<>\"']+)$")


def _block(paragraph: str) -> str:
    paragraph = paragraph.strip()
    # "See the invitation: https://..." on its own: a button with those words.
    labelled = _LABELLED_URL.match(paragraph)
    if labelled:
        return _button(labelled.group("url"), labelled.group("label").strip())
    if _URL.fullmatch(paragraph):
        return _button(paragraph, _button_label(paragraph))
    lines = "<br>".join(_linkify(line) for line in paragraph.splitlines())
    return f'<p style="margin:0 0 16px;font-size:15px;line-height:1.6;color:{INK}">{lines}</p>'


def _button(address: str, label: str) -> str:
    """A link as a button, with the address underneath for mail apps that block buttons."""
    url = html.escape(address, quote=True)
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin:8px 0 22px">'
        f'<tr><td style="border-radius:999px;background:{BLUE}">'
        f'<a href="{url}" style="display:inline-block;padding:13px 26px;font-weight:700;font-size:15px;'
        f'color:#ffffff;text-decoration:none;border-radius:999px">{html.escape(label)}</a>'
        '</td></tr></table>'
        f'<p style="margin:0 0 18px;font-size:12px;line-height:1.5;color:{MUTED}">'
        'If the button does not work, copy this address into your browser:<br>'
        f'<a href="{url}" style="color:{MUTED};word-break:break-all">{html.escape(address)}</a></p>')


def render_email_html(subject: str, body: str, base_url: str = "") -> str:
    """The HTML email for a plain-text `body`, branded, with the logo from `base_url`."""
    base = (base_url or "https://warrioriq.eu").rstrip("/")
    logo = html.escape(f"{base}/static/warrioriq-logo-96.png", quote=True)
    site = html.escape(base.split("://", 1)[-1])
    paragraphs = [part for part in re.split(r"\n\s*\n", body.strip()) if part.strip()]
    content = "".join(_block(part) for part in paragraphs)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>{html.escape(subject)}</title></head>
<body style="margin:0;padding:0;background:#eef1f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#eef1f6">
<tr><td align="center" style="padding:28px 12px">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:560px;background:#ffffff;border-radius:18px;overflow:hidden">
<tr><td style="background:#0b0f17;padding:22px 28px">
<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>
<td style="vertical-align:middle"><img src="{logo}" width="40" height="40" alt="WarriorIQ" style="display:block;border:0;border-radius:10px"></td>
<td style="vertical-align:middle;padding-left:12px;font-size:19px;font-weight:800;letter-spacing:.04em;color:#ffffff">WARRIORIQ</td>
</tr></table></td></tr>
<tr><td style="height:3px;background:linear-gradient(90deg,#5aa9ff,#ff5d68);background-color:{BLUE};font-size:0;line-height:0">&nbsp;</td></tr>
<tr><td style="padding:30px 28px 12px">
<h1 style="margin:0 0 18px;font-size:21px;line-height:1.3;color:{INK}">{html.escape(subject)}</h1>
{content}
</td></tr>
<tr><td style="padding:18px 28px 26px;border-top:1px solid #e6e9ef;font-size:12px;line-height:1.6;color:{MUTED}">
WarriorIQ &middot; fight video analysis for combat sports<br>
<a href="{html.escape(base, quote=True)}" style="color:{MUTED}">{site}</a> &middot; This is an automatic message about your account.
</td></tr>
</table>
</td></tr></table>
</body></html>"""
