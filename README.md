# Ontology-based prediction of cancer driver genes: predCAN

## Datasets

We used the driver genes from and [IntoGen](https://www.intogen.org/search) other genes from [cellular Phenotype Database](https://www.ebi.ac.uk/fg/sym)

## Dependencies

To install python dependencies run: `pip install -r requirements.txt`

## Prediction of candicate cancer driver genes workflow

We integrate the annotates from Gene ontology (GO), Cellular Microscopy Phenotype Ontology (CMPO) and Mammalian Phenotype ontology (MP) using [OPA2Vec](https://github.com/bio-ontology-research-group/opa2vec)

We used generated embeddings to test each ontology individually and evaluate their performance (AUC and F-score). Then considering the genes in which they have complete representation in GO, CMPO and MP.

```
python OPA2Vec_Prediction.py "filename"
```

And merging all three ontologies by running `mergeOntologies.groovy` to have `outont.owl`.

```
groovy mergeOntologies.groovy
```

As a result, we predict 112 new candidate driver genes within 20 cancer type `Predicted112candidateDriverGenes.txt`

## Validation on two-cohorts

Following GATK pipline with MuTect2 in tumor-only mode by running `VariantsCall-TumorOnlyMode.sh` as a job with specifiying the folder name of the VCFs files:

```
sbatch VariantsCall-TumorOnlyMode.sh
```

## A- First analysis (count the mutations)

Start by running Annovar `annovarscript.sh`:

```
chmod +x annovarscript.sh
./annovarscript.sh
```

And run `combinedhist.r` to plot the figures with trying to adjust `limits` parameter.

## B- Second Analysis (pathogenicity test)

Start with `PrepareU-test.sh` (it will run Annovar, but independent from the previous test):

```
chmod +x PrepareU-test.sh
./PrepareU-test.sh
```

Then, `ranksumtest.py` to compute different 7 p-value scores and it needs those specific cancer type related files (all-driver and predicteddriver):

```
python ranksumtest.py
```

## Final notes

For any comments or help needed with how to run the scripts, please send an email to: sara.althubaiti@kaust.edu.sa
