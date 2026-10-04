"""The Terms and Conditions shown at install time, and the record that they were accepted.

Bump ``TERMS_VERSION`` when the text changes in a way users must agree to again: the app asks
anyone who accepted an older version once more.
"""

from localdoc_finder.core.store.sqlite import StateDb

TERMS_VERSION = "1"
TERMS_ACCEPTED_KEY = "terms_accepted_version"

TERMS_TITLE = "Terms and Conditions"
TERMS_TEXT = """\
LocalDoc Finder - Terms and Conditions

Please read these terms. By choosing "I accept" you agree to them. If you do not agree, \
close this window and LocalDoc Finder will not run.

1. What LocalDoc Finder does
LocalDoc Finder indexes the files in the folders you choose so that you can search them by \
meaning and ask questions about them. Indexing, searching and chat run on this PC. Nothing \
is sent to the publisher of LocalDoc Finder: there is no account, no telemetry and no \
analytics.

2. Your files and your responsibility
You decide which folders are indexed. LocalDoc Finder reads those files and stores derived \
data (text excerpts, embeddings, thumbnails and a file list) on this PC, in its data \
folder. You are responsible for having the right to index and process the files you point \
it at, and for protecting this PC and its data folder. Files that look like secrets \
(passwords, keys, tokens) are skipped by default, but no filter is perfect; check your \
folder choices.

3. Third-party software and models
LocalDoc Finder uses Ollama and AI models that are downloaded from the internet, with your \
permission, during setup or when you add a model. Ollama and every model are provided by \
third parties under their own licences and terms, which you accept when you use them. \
LocalDoc Finder does not own, control or warrant them. Downloads can be large and use your \
disk space and bandwidth.

4. Optional cloud providers
Cloud providers are off unless you add one in Settings. If you do, the text of the \
questions you ask and the excerpts of documents needed to answer them are sent to that \
provider, under that provider's terms and prices. Government and financial identifiers are \
masked and files that look like secrets are never sent, but you remain responsible for \
what you choose to send. Your API keys are kept in Windows Credential Manager.

5. Updates
LocalDoc Finder may check for new versions on the release page of the project and download \
them. You can turn this off in Settings, under Updates.

6. AI answers can be wrong
Answers, summaries, rankings and matches are produced by AI models and can be incomplete \
or incorrect. Check them against the source documents (each answer cites them) before you \
rely on them, especially for legal, medical, financial or safety decisions.

7. No warranty
LocalDoc Finder is provided "as is" and "as available", without warranty of any kind, express \
or implied, including fitness for a particular purpose, accuracy and non-infringement.

8. Limitation of liability
To the fullest extent the law allows, the authors and publisher of LocalDoc Finder are not \
liable for any loss or damage (including lost data, lost profits or indirect damages) \
arising from your use of it, or from third-party software and models it uses. Keep \
backups of your files.

9. Changes and ending
These terms may change in a new version; you will be asked to accept the new text. You \
can stop at any time by uninstalling LocalDoc Finder; its data folder can then be deleted.
"""


def terms_accepted(state: StateDb) -> bool:
    """True once the user accepted the current version of the terms."""
    return state.get_meta(TERMS_ACCEPTED_KEY) == TERMS_VERSION


def accept_terms(state: StateDb) -> None:
    state.set_meta(TERMS_ACCEPTED_KEY, TERMS_VERSION)
