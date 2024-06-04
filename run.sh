for i in mlp mim ablation linear neuro sachs
do
    python parse.py $i
done

for i in {1..5}
do
    python parse.py dream$i
done
python dream.py

python plot.py SIM
python plot.py REAL
python analysis.py