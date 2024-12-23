import json
from getpass import getpass
from typing import Dict, Any, List

import jsonlines
from datasets import DatasetDict, load_dataset
import os


def repair_system_acts(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare the dialogues for upload to hugging face. This means normalizing the data to a consistent structure format
    (currently, only issues were with system_acts, which I don't use)
    """
    for turn in dialogue['dialogue']:
        # TODO: may be nice to actually share a version of this dataset that includes properly formatted system_acts
        del turn['system_acts']
    return dialogue


if __name__ == '__main__':
    # read each data file and write to a corresponding jsonl file
    for split in ('train', 'dev', 'test'):
        if not os.path.exists(f'data/mwz2.4/{split}_dials.json'):
            raise FileNotFoundError("dataset files not prepared properly. See README (run create_data.py)")
        with open(f'data/mwz2.4/{split}_dials.json', 'r') as f:
            dialogues: List[Dict[str, Any]] = json.load(f)
        dialogues = [repair_system_acts(d) for d in dialogues]
        os.makedirs('data/jsonl/mwz2.4/', exist_ok=True)
        with jsonlines.open(f'data/jsonl/mwz2.4/{split}_dials.jsonl', mode='w') as writer:
            writer.write_all(dialogues)

    # then point to the folder and load_dataset
    dataset: DatasetDict = load_dataset('data/jsonl/mwz2.4')
    api_token = os.environ.get("HF_API_TOKEN") or getpass(prompt="Paste huggingface API token:")
    dataset.push_to_hub("Brendan/multi_woz_v24", private=True, token=api_token)