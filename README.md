# PoliticalTweetAnalysis
General Analysis of tweets regarding the United States Presidential Election and BREXIT.


## Installation

First install Python dependencies via:

	pip install -r requirements.txt


Then, download corpora for NLTK (e.g., stopwords) by doing:

	python 
	import nltk
	nltk.download()

and selecting either `all-corpora` from the GUI that pops up, or (better option!) browsing to the `corpora` tab and selecting only the `stopwords` corpus.

From here, you should be able to run

	python load_data.py

... Hopefully.