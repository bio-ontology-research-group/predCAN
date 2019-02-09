#!/bin/bash

#SBATCH --mem 180Gb # memory pool for all cores
#SBATCH --time 90:00:00 # time, specify max time allocation
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=sara.althubaiti@kaust.edu.sa  
#SBATCH --cpus-per-task=10
#SBATCH --job-name=D508_L001
#SBATCH --output=/encrypted/e3000/gatkwork/D508_L001.log

module load gatk/4.0.1.1
module load samtools
module load picard/2.17.6
module load bwa/0.7.17/gnu-6.4.0

path_to_ref='/encrypted/e3000/gatkwork/ref'
additional_sample_files='/encrypted/e3000/gatkwork/Bamfiles'
sample_name='D508_L001'
Vname='Ready'
annotaion='DepthPerSampleHC' # add DP in FORMAT column
annotaion1='Coverage' # add DP in INFO column
annotaion2='DepthPerAlleleBySample' # add AD in FORMAT

gatk Mutect2 \
-R "$path_to_ref"/human_g1k_v37.fasta \
-I "$additional_sample_files/$sample_name"_recal_reads.bam \
-tumor "LIBRARY_NAME=Solexa-272222" \
-A "$annotaion" \
-A "$annotaion1" \
-A "$annotaion2" \
--germline-resource "$path_to_ref"/af-only-gnomad.raw.sites.vcf.gz \
--af-of-alleles-not-in-resource 0.001 \
--disable-read-filter MateOnSameContigOrNoMappedMateReadFilter \
--output "$additional_sample_files/$sample_name"_step2part1_.vcf.gz

# Just one time
# gatk SelectVariants \
# -R "$path_to_ref"/human_g1k_v37.fasta \
# -V "$path_to_ref"/gnomad.exomes.r2.0.2.nsites.vcf \
# --select-type-to-include SNP \
# -O "$additional_sample_files/$sample_name"_SVarient_.vcf

gatk GetPileupSummaries \
-I "$additional_sample_files/$sample_name"_recal_reads.bam \
-V "$path_to_ref/$Vname"_SVarient_.vcf \
-O "$additional_sample_files/$sample_name"_PileupSummaries_.table

gatk CalculateContamination \
-I "$additional_sample_files/$sample_name"_PileupSummaries_.table \
-O "$additional_sample_files/$sample_name"_calculatecontamination_.table

gatk FilterMutectCalls \
-V "$additional_sample_files/$sample_name"_step2part1_.vcf.gz \
--contamination-table "$additional_sample_files/$sample_name"_calculatecontamination_.table \
-O "$additional_sample_files/$sample_name"_oncefiltered_.vcf.gz