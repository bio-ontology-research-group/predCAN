# the path to input/samples folder
path_to_vcf='/encrypted/e3000/gatkwork/vcf'

# the path to Results folder
path_to_output='/encrypted/e3000/gatkwork/Results'

# the path to annovar folder
path_to_annovar='/encrypted/e3000/gatkwork/annovar'

for entry in $path_to_vcf/*
do

fbname=$(basename "$entry" .vcf)

# start annotate the vcf file
perl "$path_to_annovar"/table_annovar.pl "$path_to_vcf/$fbname".vcf humandb/ -buildver hg19 -out "$path_to_output/OutputAnnoFile_$fbname" -remove -protocol refGene,cytoBand,exac03,avsnp147,dbnsfp30a -operation g,r,f,f,f -nastring . -vcfinput

# Take the needed columns (7, 17, 30, 44, 47, 56, 59, 62, 68 and 79) -> start, end and gene name
awk '{print $12,$20,$21,$25,$29,$31,$33,$37,$38,$7}' "$path_to_output/OutputAnnoFile_$fbname".hg19_multianno.txt > "$path_to_output/Output_$fbname".txt

#sed 's/\;[^;]*$//' "$path_to_output/Output_$fbname".txt > "$path_to_output/Ready_$fbname".txt

rm "$path_to_output"/*.vcf
rm "$path_to_output"/*.avinput
rm "$path_to_output/OutputAnnoFile_$fbname".hg19_multianno.txt
#rm "$path_to_output/Output_$fbname".txt

# filter by Freq_allel < 0.01
awk '($1 != ".")' "$path_to_output/Output_$fbname".txt > "$path_to_output/Filtered_$fbname".txt
awk '($1 < 0.01)' "$path_to_output/Filtered_$fbname".txt > "$path_to_output/Filtered1_$fbname".txt

rm "$path_to_output/Output_$fbname".txt
rm "$path_to_output/Filtered_$fbname".txt

# remove any missing scores in any of the 7 different pathogenic scores
awk '($3 != "." && $4 != "." && $5 != "." && $6 != "." && $7 != "." && $8 != "." && $9 != ".")' "$path_to_output/Filtered1_$fbname".txt > "$path_to_output/FilteredWithoutDot_$fbname".txt

rm "$path_to_output/Filtered1_$fbname".txt

sed 's/\;[^;]*$//' "$path_to_output/FilteredWithoutDot_$fbname".txt > "$path_to_output/FilteredWithoutDot1_$fbname".txt

rm "$path_to_output/FilteredWithoutDot_$fbname".txt

done
