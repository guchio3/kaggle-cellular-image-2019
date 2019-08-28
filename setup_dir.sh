# mkdirs
mkdir configs
mkdir configs/exp_configs/
mkdir notebooks
mkdir tools
mkdir tools/utils
mkdir tools/models
mkdir mnt
# mkdir mnt/inputs/origin/
# mkdir mnt/oofs/
# mkdir mnt/logs/
# mkdir mnt/submissions/
# mkdir mnt/trained_models/

# set .gitignore
if [ -e ".gitignore" ]; then
    echo ".gitignore already exists!"
else
    touch .gitignore
    echo "tags*" >> .gitignore
    echo "mnt/" >> .gitignore
fi

git clone https://github.com/recursionpharma/rxrx1-utils tools/utils/rxrx1-utils
