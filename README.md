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

Then, download corpora for TextBlob by doing:
    
    python -m textblob.download_corpora

From here, you should be able to run

	python analyze.py

... Hopefully.

## Visualize Geo Data

To visualize where each tweet came from using Leaflet.js, run this command to create the geo_data.json file:

	python -m SimpleHTTPServer 8888

and then open visualize_geodata.html on your browswer. You should then be able to interact with a map containing all the tweets and their origin.

