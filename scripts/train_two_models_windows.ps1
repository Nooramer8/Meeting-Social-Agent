# Run from the project root: D:\meeting_social_agent
# This trains the two "our training" models after you create train/eval JSONL files.

.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-training.txt
python training\validate_training_data.py
python training\train_our_two_models.py --asr_epochs 3 --sum_epochs 3 --batch_size 2
