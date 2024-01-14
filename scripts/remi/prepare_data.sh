> dict/additional_tokens/remi_new.txt

export PYTHONPATH='.'

# melody
python prepare_data/new/remi/main.py --dataset=pop909 --task=melody

# # velocity
python prepare_data/new/remi/main.py --dataset=pop909 --task=velocity

# # composer
python prepare_data/new/remi/main.py --dataset=pianist8 --task=composer

# emotion
python prepare_data/new/remi/main.py --dataset=emopia --task=emotion

python prepare_data/new/remi/main.py --dataset=pop1k7

python prepare_data/new/remi/main.py --dataset=ASAP

# save dict file
python prepare_data/new/remi/config.py