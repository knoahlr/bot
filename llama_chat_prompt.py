# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

import time

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    input_str = " Start conversation pls "
    while True:
        start = time.time()
        current_input = input(f"{input_str}")
        time_to_input = time.time()-start
        prompt_input_dict : List[str] = [f"{current_input}",]
        results = generator.text_completion(
            prompt_input_dict,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        time_to_generate = time.time()-start
        time_to_input_str = time.strftime("%H:%M:%S", time.gmtime(time_to_input))
        time_to_generate_str = time.strftime("%H:%M:%S", time.gmtime(time_to_generate))
        print(f"\n\tTime to input:{time_to_input_str}\n\tTime to response:{time_to_generate_str}")
        print(f"Llama:\n\t{results[0]['generation']}")
        input_str = "User:\n\t"

if __name__ == "__main__":
    fire.Fire(main)
