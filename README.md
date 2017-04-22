# TensorFlow data dealer

This contains utilities to deal with TF data format from your raw data:

* Scripts to convert raw data (e.g. text in CSV) into TFRecords
* API to the generated records with TensorFlow pipelines.

### Conversion scripts available

#### text_to_tfrecord

The input is a CSV file containing text to be encoded in different columns. There are two options of encoding:

* Word based codes: one index per word.
* Character based codes: one index per character, so `N[i]` indexes for word `i`.

There is the possibility to specify excluded columns as input arguments to be ignored in the encoding process. Labels can also be included by specifying the label column index in the arguments.

For example we may have a CSV file `data.csv` with the contents:

```
id, sentence, label
0, "The rainy day", 0
1, "The sunny valley", 0
2, "The rich man has money", 1
...
```

The utility can encode the text at column `sentence`, ignore the column `id` and get the `label` (integer always) by executing:

```
python text_to_tfrecords.py --csv_file data.csv --exclude_csv_fields 0 --label_idx 2 --out_prefix example --do-chars
```

This will create a file `example_data.tfrecords` where words, chars and labels are encoded in sequential examples.
