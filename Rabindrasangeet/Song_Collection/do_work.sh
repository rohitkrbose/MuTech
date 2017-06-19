for d in */ ; do
	vd="\""$d"\""
	C="cd "$vd""
	echo $d
	eval $C
	mv ../test.sh .
	c="mv ../$1 ."
	eval $c
	c="bash test.sh $1"
	eval $c	
	mv test.sh ../
	c="mv $1 ../"
	eval $c
	cd ..
done
