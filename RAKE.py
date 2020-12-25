
!pip install rake-nltk
from pathlib import Path
text=Path('/content/drive/My Drive/Colab Notebooks/amazon_small.txt').read_text()
from rake_nltk import Rake
# Uses stopwords for english from NLTK, and all puntuation characters by
# default
r = Rake()
# Extraction given the text.
r.extract_keywords_from_text(text)
# To get keyword phrases ranked highest to lowest with score
r.get_ranked_phrases_with_scores()[0:50]
