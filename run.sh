for i in mlp mim ablation linear
do
    python parse.py $i
done

for i in {1..5}
do
    python parse.py dream$i
done
python dream.py
python parse.py neuro
python parse.py sachs

python plot.py SIM
python plot.py REAL
python analysis.py