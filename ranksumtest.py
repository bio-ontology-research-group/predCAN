import numpy as np
from scipy.stats import ranksums
import pandas as pd
import csv

file = pd.read_csv('merged-file.txt', header=None, skiprows=0, delim_whitespace=True)

file.columns = ['Freq_allel','dpsnp','sift','polyphen','mutas','muaccessor','fathmm','vest3','CADD','geneName']

df = file.drop_duplicates(keep=False)

################## START ###################

# calculate ranksums for SIFT

sift_df = df[['geneName','sift']]

# extract all non-driver genes | sift_score
genelist = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-all-driver.tsv', header=None, skiprows=0, sep='\t')
genelist.columns = ['geneName']
#
merged_df = pd.merge(
    sift_df, genelist,
    how='outer', on=['geneName'], indicator=True, suffixes=('_foo','')).query(
        '_merge == "left_only"')

merged_df.drop(['geneName','_merge'], axis=1, inplace=True)

# extract all predicted driver genes | sift_score

genelist1 = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-predicteddriver.tsv', header=None, skiprows=0, sep='\t')
genelist1.columns = ['geneName']

merged_df1 = sift_df.merge(genelist1, how = 'inner', on = ['geneName'])

merged_df1.drop(['geneName'], axis=1, inplace=True)

# calculate p-value for ranksums with SIFT

stat, pvalue = ranksums(merged_df, merged_df1)

print(pvalue)

#################### POLYPHEN ###################

# calculate ranksums for POLYPHEN

polyphen_df = df[['geneName','polyphen']]

# extract all non-driver genes | sift_score
genelist = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-all-driver.tsv', header=None, skiprows=0, sep='\t')
genelist.columns = ['geneName']
#
merged_df = pd.merge(
    polyphen_df, genelist,
    how='outer', on=['geneName'], indicator=True, suffixes=('_foo','')).query(
        '_merge == "left_only"')

merged_df.drop(['geneName','_merge'], axis=1, inplace=True)

# extract all predicted driver genes | polyphen_score

genelist1 = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-predicteddriver.tsv', header=None, skiprows=0, sep='\t')
genelist1.columns = ['geneName']

merged_df1 = polyphen_df.merge(genelist1, how = 'inner', on = ['geneName'])

merged_df1.drop(['geneName'], axis=1, inplace=True)

# calculate p-value for ranksums with polyphen

stat, pvalue = ranksums(merged_df, merged_df1)

print(pvalue)

#################### MutationTaster ###################

# calculate ranksums for MutationTaster

mutas_df = df[['geneName','mutas']]

# extract all non-driver genes | MutationTaster_score
genelist = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-all-driver.tsv', header=None, skiprows=0, sep='\t')
genelist.columns = ['geneName']
#
merged_df = pd.merge(
    mutas_df, genelist,
    how='outer', on=['geneName'], indicator=True, suffixes=('_foo','')).query(
        '_merge == "left_only"')

merged_df.drop(['geneName','_merge'], axis=1, inplace=True)

# extract all predicted driver genes | MutationTaster_score

genelist1 = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-predicteddriver.tsv', header=None, skiprows=0, sep='\t')
genelist1.columns = ['geneName']

merged_df1 = mutas_df.merge(genelist1, how = 'inner', on = ['geneName'])

merged_df1.drop(['geneName'], axis=1, inplace=True)

# calculate p-value for ranksums with MutationTaster

stat, pvalue = ranksums(merged_df, merged_df1)

print(pvalue)

#################### Mutationassessor ###################

# calculate ranksums for Mutationassessor

muaccessor_df = df[['geneName','muaccessor']]

# extract all non-driver genes | Mutationassessor_score
genelist = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-all-driver.tsv', header=None, skiprows=0, sep='\t')
genelist.columns = ['geneName']
#
merged_df = pd.merge(
    muaccessor_df, genelist,
    how='outer', on=['geneName'], indicator=True, suffixes=('_foo','')).query(
        '_merge == "left_only"')

merged_df.drop(['geneName','_merge'], axis=1, inplace=True)

# extract all predicted driver genes | Mutationassessor_score

genelist1 = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-predicteddriver.tsv', header=None, skiprows=0, sep='\t')
genelist1.columns = ['geneName']

merged_df1 = muaccessor_df.merge(genelist1, how = 'inner', on = ['geneName'])

merged_df1.drop(['geneName'], axis=1, inplace=True)

# calculate p-value for ranksums with Mutationassessor

stat, pvalue = ranksums(merged_df, merged_df1)

print(pvalue)

#################### fathmm ###################

# calculate ranksums for fathmm

fathmm_df = df[['geneName','fathmm']]

# extract all non-driver genes | fathmm_score
genelist = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-all-driver.tsv', header=None, skiprows=0, sep='\t')
genelist.columns = ['geneName']
#
merged_df = pd.merge(
    fathmm_df, genelist,
    how='outer', on=['geneName'], indicator=True, suffixes=('_foo','')).query(
        '_merge == "left_only"')

merged_df.drop(['geneName','_merge'], axis=1, inplace=True)

# extract all predicted driver genes | fathmm_score

genelist1 = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-predicteddriver.tsv', header=None, skiprows=0, sep='\t')
genelist1.columns = ['geneName']

merged_df1 = fathmm_df.merge(genelist1, how = 'inner', on = ['geneName'])

merged_df1.drop(['geneName'], axis=1, inplace=True)

# calculate p-value for ranksums with fathmm

stat, pvalue = ranksums(merged_df, merged_df1)

print(pvalue)

#################### VEST3 ###################

# calculate ranksums for VEST3

vest3_df = df[['geneName','vest3']]

# extract all non-driver genes | VEST3_score
genelist = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-all-driver.tsv', header=None, skiprows=0, sep='\t')
genelist.columns = ['geneName']
#
merged_df = pd.merge(
    vest3_df, genelist,
    how='outer', on=['geneName'], indicator=True, suffixes=('_foo','')).query(
        '_merge == "left_only"')

merged_df.drop(['geneName','_merge'], axis=1, inplace=True)

# extract all predicted driver genes | VEST3_score

genelist1 = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-predicteddriver.tsv', header=None, skiprows=0, sep='\t')
genelist1.columns = ['geneName']

merged_df1 = vest3_df.merge(genelist1, how = 'inner', on = ['geneName'])

merged_df1.drop(['geneName'], axis=1, inplace=True)

# calculate p-value for ranksums with VEST3

stat, pvalue = ranksums(merged_df, merged_df1)

print(pvalue)

#################### CADD ###################

# calculate ranksums for CADD

CADD_df = df[['geneName','CADD']]

# extract all non-driver genes | CADD_score
genelist = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-all-driver.tsv', header=None, skiprows=0, sep='\t')
genelist.columns = ['geneName']
#
merged_df = pd.merge(
    CADD_df, genelist,
    how='outer', on=['geneName'], indicator=True, suffixes=('_foo','')).query(
        '_merge == "left_only"')

merged_df.drop(['geneName','_merge'], axis=1, inplace=True)

# extract all predicted driver genes | CADD_score

genelist1 = pd.read_csv('/encrypted/e3000/gatkwork/COREAD-ESCA-predicteddriver.tsv', header=None, skiprows=0, sep='\t')
genelist1.columns = ['geneName']

merged_df1 = CADD_df.merge(genelist1, how = 'inner', on = ['geneName'])

merged_df1.drop(['geneName'], axis=1, inplace=True)

# calculate p-value for ranksums with CADD

stat, pvalue = ranksums(merged_df, merged_df1)

print(pvalue)