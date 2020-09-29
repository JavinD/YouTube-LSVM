# YouTube-LSVM
Emotion Analysis of YouTube Videos with LSVM

Projek ini memanfaatkan algoritma LSVM untuk melakukan kategorisasi terhadap video-video pada website YouTube.com.

# 1. YouTube Dataset Crawling

Menggunakan library pyTube3 dan requests untuk export subtitles dari video menjadi .txt dalam jumlah yang besar.

# 2. Emotion Analysis

Batch file yang berisis subtitle akan dianalisa.
a. Preprocess dataset
b. LSVM dari SGDClassifier menggunakan library sklearn
c. Melakukan kategorisasi 'emotions'
d. Check metrics

