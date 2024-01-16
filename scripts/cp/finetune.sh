export PYTHONPATH="."

# melody
python3 MidiBERT/cp/finetune.py --tokenization=cp --task=melody --name=default --num_workers=4 --batch_size=4

# velocity
python3 MidiBERT/cp/finetune.py --tokenization=cp --task=velocity --name=default --num_workers=4 --batch_size=4

# composer
python3 MidiBERT/cp/finetune.py --tokenization=cp --task=composer --name=default --num_workers=4 --batch_size=4

# emotion
python3 MidiBERT/cp/finetune.py --tokenization=cp --task=emotion --name=default --num_workers=4 --batch_size=4

