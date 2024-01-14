> dict/additional_tokens/structured_new.txt

export PYTHONPATH='.'

# melody
python prepare_data/new/structured/main.py --dataset=pop909 --task=melody

# velocity
python prepare_data/new/structured/main.py --dataset=pop909 --task=velocity

# composer
python prepare_data/new/structured/main.py --dataset=pianist8 --task=composer

# emotion
python prepare_data/new/structured/main.py --dataset=emopia --task=emotion

python prepare_data/new/structured/main.py --dataset=pop1k7

python prepare_data/new/structured/main.py --dataset=ASAP

# save dict file
python prepare_data/new/structured/config.py