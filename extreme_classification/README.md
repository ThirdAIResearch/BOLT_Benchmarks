## Dataset Downloads

You can download the extreme classification datasets from the following webpage:
http://manikvarma.org/downloads/XC/XMLRepository.html

The datasets to download are (using the BoW Features link):
Amazon-670K	
Delicious-200K	
WikiSeeAlsoTitles-350K

For e.g. Amazon 670k, download the zip file and move it in to this directory, then run the following commands:

```
unzip Amazon670k.bow.zip
rm Amazon670K.bow.zip
mv Amazon670K.bow Amazon670k
```


## Reproducing Experiments
Reproducing the BOLT extreme classification results is simple. Just run the corresponding script. 

For e.g. Amazon 670k, run

```
python3 bolt_amazon670k.py
```