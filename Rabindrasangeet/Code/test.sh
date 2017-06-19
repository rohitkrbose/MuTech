mkdir Copy
for audx in *.mp3
do
	vx="\""$audx"\""
	for audy in *.mp3
	do
		vy="\""$audy"\""
		if [ "$audx" != "$audy" ]; then
			co="python $1 $vx $vy >> ../Records_$1.csv"
			eval $co
		fi
	done
	z="mv ""$vx"" Copy/""$vx"
	eval $z  
done
cd Copy
mv *.* ../
cd ..
rmdir Copy