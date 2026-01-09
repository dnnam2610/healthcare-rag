import re

def clean_word(w: str) -> str:
    """
    - Lowercase
    - Remove leading/trailing punctuation
    - Keep Vietnamese characters
    """
    w = w.lower()

    # Remove numbering like "1.", "2.", "10."
    if re.fullmatch(r"\d+\.", w):
        return ""

    # Remove pure numbers
    if w.isdigit():
        return ""

    # Remove non-letter characters (keep Vietnamese unicode)
    w = re.sub(r"[^\wàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", "", w)

    return w

def preprocess_text(text: str) -> str:
    """
    Clean raw text before word segmentation
    """
    text = text.replace("\n", " ").replace("-", " ")

    words = text.split()
    words = [clean_word(w) for w in words]
    words = [w for w in words if w]  # remove empty tokens

    return " ".join(words)
