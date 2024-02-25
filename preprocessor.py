import argparse
from datasets import Dataset
import json
import os
import pandas as pd
import typing


_FORBIDDEN_MESSAGE_FIELDS = {
    'media_type', 'mime_type', 'edited', 'photo',
    'location_information', 'poll', 'contact_information'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--telegram_data', type=str,
        default=os.path.join('raw_data', 'result.json'),
        help='Path to the exported telegram data (*.json)',
    )
    parser.add_argument(
        '--actor_to_alias', type=str,
        default=os.path.join('raw_data', 'actor_to_alias.json'),
        help='Path to the alias map (*.json)',
    )
    parser.add_argument(
        '--output', type=str,
        default='ift_dataset',
        help='Path to the output dataset',
    )
    parser.add_argument(
        '--cooling_time', type=int,
        default=30,
        help='Conversation cooling time (mins)',
    )
    parser.add_argument(
        '--max_replicas', type=int,
        default=10,
        help='Max replicas per conversation',
    )
    parser.add_argument(
        '--min_replicas', type=int,
        default=3,
        help='Min replicas per conversation',
    )
    return parser.parse_args()


def run_preprocessing(
    raw_dataset: typing.Dict[str, typing.Any],
    actor_to_alias: typing.Dict[str, str],
    output: str,
    conversation_cooling_time_min: int = 30,
    max_replics_per_conversation: int = 10,
    min_replics_per_conversation: int = 3,
):
    messages = raw_dataset['messages']
    print(f'Total messages: {len(messages)}')

    def _filter_predicate(telegram_message):
        if telegram_message.get('type') != 'message':
            return False
        if any(x in telegram_message for x in _FORBIDDEN_MESSAGE_FIELDS):
            return False

        actor = telegram_message.get('from')
        if actor is None:
            return False
        if actor not in actor_to_alias:
            return False

        text = telegram_message.get('text')
        if not isinstance(text, str):
            return False

        return True

    filtered_messages = [
        x for x in messages if _filter_predicate(x)
    ]
    print(f'Filtered messages: {len(filtered_messages)}')

    conversations = []
    current_conversation = []
    current_actor = None
    current_stamp = None
    for m in filtered_messages:
        new_stamp = int(m['date_unixtime'])
        new_actor = m['from']
        new_text = m['text']

        large_pause = current_stamp is not None and new_stamp > (current_stamp + conversation_cooling_time_min * 60)
        same_actor = current_actor == new_actor

        if not large_pause and same_actor:
            new_replic = f'{current_conversation[-1]}\n{new_text}'
            current_conversation.pop()
            current_conversation.append(new_replic)
            current_stamp = new_stamp
            continue
            

        if len(current_conversation) >= max_replics_per_conversation or large_pause:
            if len(current_conversation) >= min_replics_per_conversation:
                conversations.append(current_conversation)
            current_conversation = []
            current_actor = None
            current_stamp = None

        current_conversation.append(f'{actor_to_alias[new_actor]}: {new_text}')

        current_actor = new_actor
        current_stamp = new_stamp

    print(f'Total conversations: {len(conversations)}')
    print(f'Total replics: {sum(len(x) for x in conversations)}')

    dataset = Dataset.from_pandas(
        pd.DataFrame(
            {'prompt': ['\n'.join(x) for x in conversations]}
        )
    )
    dataset.save_to_disk(output)
    print(f'Output dataset saved @ {output}')


if __name__ == '__main__':
    args = parse_args()

    with open(args.telegram_data, 'rt') as f:
        raw_dataset = json.load(f)

    with open(args.actor_to_alias, 'rt') as f:
        actor_to_alias = json.load(f)

    run_preprocessing(
        raw_dataset=raw_dataset,
        actor_to_alias=actor_to_alias,
        output=args.output,
        conversation_cooling_time_min=args.cooling_time,
        max_replics_per_conversation=args.max_replicas,
        min_replics_per_conversation=args.min_replicas,
    )
