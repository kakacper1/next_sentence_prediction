# HOW TO: Data processing pipeline from wikipedia dump to train, dev, and test datasets.

1) Download wikipedia dump from "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2".
In case of smaller datasets, go to "https://dumps.wikimedia.org/enwiki/latest/" and download e.g. "enwiki-latest-pages-articles1.xml-p10p30302.bz2".
    1) extract the .bz2 file to `resources/data/` which will create e.g. the following file `resources/data/enwiki-latest-pages-articles3.xml-p88445p200507`

2) Process the dump to get files in sentence-per-line format with empty lines denoting the end of a paragraph
    1) Create `enwiki_out` directory inside `resources/data`
    2) cd to `helpers/wikiextractor/`
    3) Run: `python WikiExtractor.py ../../resources/data/enwiki-latest-pages-articles3.xml-p88445p200507 -o ../../resources/data/enwiki_out/` which will run the WikiExtractor and output the resulting files to `resources/data/enwiki_out/`

3) Do additional data processing to remove special characters
    1) Open `notebooks/data_preprocessing.ipynb` notebook. Go to the section `Prepare wikipedia files in one-sentence-per-line format`, specify the files you want to process (e.g. `../resources/data/enwiki_out/B*/wiki_*` to process all wiki files in folders starting with "B")

4)
    **a)** Create .tsv files in 3-column format (label, sentence_a, sentence_b) for data distribution of *50% true* next sentences and *50% random* next sentences.
    1) Download and extract Bert vocabulary file: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    
    2) Run
        ```
        for f in <ABSOLUTE-PATH-TO-THIS-DATA-GENERATION-FOLDER>/resources/data/wiki_preprocessed/*; do
          python <ABSOLUTE-PATH-TO-THIS-DATA-GENERATION-FOLDER>/bert/create_pretraining_data_3_col_format.py  --input_file="$f"   --output_file=<ABSOLUTE-PATH-TO-THIS-DATA-GENERATION-FOLDER>/resources/data/Gutenberg/pokus.tfrecord   --vocab_file=<ABSOLUTE-PATH-TO-VOCAB-FILE>/uncased_L-12_H-768_A-12/vocab.txt   --do_lower_case=True   --max_seq_length=128   --max_predictions_per_seq=20   --masked_lm_prob=0.15   --random_seed=12345   --dupe_factor=5 --filename="${f}.csv"
        done
        ```
    3) Move the created .csv files to new folder. `mkdir basic_50_50`. `mv B*.csv basic_50_50/`

    **b)** Create .tsv files in 4-column format (label, sentence_a, sentence_b, next_sent_type) for data distribution of *33% true* next sentences, *33% random*, and *33% swapped* sentences.
    1) Download and extract Bert vocabulary file: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    2) Run
        ```
        for f in <ABSOLUTE-PATH-TO-THIS-DATA-GENERATION-FOLDER>/resources/data/wiki_preprocessed/*; do
          python <ABSOLUTE-PATH-TO-THIS-DATA-GENERATION-FOLDER>/bert/create_pretraining_data_4_col_swapped_format.py  --input_file="$f"   --output_file=<ABSOLUTE-PATH-TO-THIS-DATA-GENERATION-FOLDER>/resources/data/Gutenberg/pokus.tfrecord   --vocab_file=<ABSOLUTE-PATH-TO-VOCAB-FILE>/uncased_L-12_H-768_A-12/vocab.txt   --do_lower_case=True   --max_seq_length=128   --max_predictions_per_seq=20   --masked_lm_prob=0.15   --random_seed=12345   --dupe_factor=5 --filename="${f}.csv"
        done
        ```
    3) Move the created .csv files to new folder. `mkdir swapped_33_33_33`. `mv *.csv swapped_33_33_33/`

5) Create *train*, *dev*, and *test* datasets
    1) Go to directory where your .csv files are (`swapped_33_33_33`, or `basic_50_50`). Run `cat *.csv > 33_33_33_all.tsv` to create one big file with all sentence pairs
    2) If we want to have 100k sentence pairs in train, 10k in dev, and 10k in test set, we can run the following commands (given that the size of all our data is <=120000).
    3) Get train set by running `sed -n '1,100000p' 33_33_33_all.tsv > train.tsv`, dev by running `sed -n '100001,110000p' 33_33_33_all.tsv > dev.tsv`, and test by running `sed -n '110001,120000p' 33_33_33_all.tsv > test.tsv`.


