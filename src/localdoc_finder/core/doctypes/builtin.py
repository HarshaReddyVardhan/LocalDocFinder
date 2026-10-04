"""Built-in document types. Each is a name; the evidence comes from ``DocTypeSettings.rules``.

To add a type: add a class like these in any new module of this package (or a rule entry in
settings plus a one-line class here) and decorate it with ``@register_doctype``.
"""

from localdoc_finder.core.doctypes.base import RuleBasedClassifier, register_doctype


@register_doctype("resume")
class ResumeClassifier(RuleBasedClassifier):
    name = "resume"
    priority = 10


@register_doctype("cover_letter")
class CoverLetterClassifier(RuleBasedClassifier):
    name = "cover_letter"
    priority = 20


@register_doctype("jd")
class JobDescriptionClassifier(RuleBasedClassifier):
    name = "jd"
    priority = 30


@register_doctype("invoice")
class InvoiceClassifier(RuleBasedClassifier):
    name = "invoice"
    priority = 40


@register_doctype("paper")
class PaperClassifier(RuleBasedClassifier):
    name = "paper"
    priority = 50


@register_doctype("plan")
class PlanClassifier(RuleBasedClassifier):
    name = "plan"
    priority = 60


@register_doctype("notes")
class NotesClassifier(RuleBasedClassifier):
    name = "notes"
    priority = 70
