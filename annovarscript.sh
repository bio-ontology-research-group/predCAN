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

# Take the needed columns (2, 3 and 7) -> start, end and gene name
awk '{print $2,$3,$7}' "$path_to_output/OutputAnnoFile_$fbname".hg19_multianno.txt > "$path_to_output/Output_$fbname".txt

sed 's/\;[^;]*$//' "$path_to_output/Output_$fbname".txt > "$path_to_output/Ready_$fbname".txt

rm "$path_to_output"/*.vcf
rm "$path_to_output"/*.avinput
rm "$path_to_output/OutputAnnoFile_$fbname".hg19_multianno.txt
rm "$path_to_output/Output_$fbname".txt

done
