## Caltech-101 dataset
The example notebook uses a ResNet-18 model trained on Caltech-101 dataset. To run the examples, download and extract all the archives from the Caltech-101 dataset ([link](https://data.caltech.edu/records/mzrjq-6wc02)) here. Once extracted, the directory structure should look like this:
```
./
    caltech-101/
        101_ObjectCategories/
            ...
        annotations/
            ...
```

Now, you can run the example notebook [`ffm.ipynb`](../ffm.ipynb) to compute FFM scores on some examples from the Caltech-101 dataset.