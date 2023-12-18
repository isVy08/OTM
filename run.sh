for i in mlp mim gp-add
do
    for j in ER SF
    do
        python visualize.py $i $j
    done
done