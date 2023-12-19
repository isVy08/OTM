rm figures/*.png
# rm output/*.pickle
# for i in mlp mim
# do
#     python parse.py $i
# done

for i in mlp mim
do
    for j in ER SF REAL
    do
        python visualize.py $i $j
    done
done

python visualize.py real REAL