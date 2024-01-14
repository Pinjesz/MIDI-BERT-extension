export PYTHONPATH="."

# melody
python3 MidiBERT/simple/eval.py --tokenization=remi --task=melody --name=default --num_workers=4 --batch_size=4

# velocity
python3 MidiBERT/simple/eval.py --tokenization=remi --task=velocity --name=default --num_workers=4 --batch_size=4

# composer
python3 MidiBERT/simple/eval.py --tokenization=remi --task=composer --name=default --num_workers=4 --batch_size=4

# emotion
python3 MidiBERT/simple/eval.py --tokenization=remi --task=emotion --name=default --num_workers=4 --batch_size=4