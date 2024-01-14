export PYTHONPATH="."

# melody
python MidiBERT/cp/eval.py --tokenization=cp --task=melody --num_workers=4 --batch_size=4

# velocity
python MidiBERT/cp/eval.py --tokenization=cp --task=velocity --num_workers=4 --batch_size=4

# composer
python MidiBERT/cp/eval.py --tokenization=cp --task=composer --num_workers=4 --batch_size=4

# emotion
python MidiBERT/cp/eval.py --tokenization=cp --task=emotion --num_workers=4 --batch_size=4