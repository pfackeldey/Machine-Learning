array=("hww120_200" "hww210_500" "hww550_3000" "DY" "WW" "VVV" "DiBoson" "WJets" "Top" "data") 
for i in "${array[@]}" 
do
	echo "RUNNING: python evaluation/keras_evaluation.py config/MSSM_HWW.yaml --tree ${i}"
	echo "Processing tree: ${i}"
	python evaluation/keras_evaluation.py config/MSSM_HWW.yaml --tree ${i}
done

