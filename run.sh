# rm figures/*.png
# rm output/*.pickle
# for i in mlp mim
# do
#     python parse.py $i
# done

# for i in {1..5}
# do
#     python parse.py dream$i
# done
# python dream.py
# python parse.py neuro
# python parse.py sachs

for i in mlp mim
do
    for j in ER SF
    do
        python visualize.py $i $j
    done
done


python visualize.py dream REAL
python visualize.py neuro REAL
python visualize.py sachs REAL