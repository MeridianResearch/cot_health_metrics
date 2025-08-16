#!/usr/bin/env python3
"""
Simple script to create the filler_texts.json file.
Run: python create_filler_texts.py
"""

import json
import os


def create_filler_texts_file():
    filler_texts_data = {
        "filler_texts": {
            "lorem_ipsum": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem.",
            "cicero_original": "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur. Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur. Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae.",
            "random_words": "apple banana cherry dog elephant forest green house island jump kitchen lamp mountain notebook ocean paper question river stone table umbrella violet water yellow zebra anchor bridge castle dragon eagle flower garden harbor ice jungle king lion mouse nature orange purple queen rainbow silver thunder valley whisper xenon yacht zigzag abundance brilliant courage dynamic elegant fantastic gracious harmony infinite journey knowledge luminous magnificent noble optimistic peaceful quality resilient serene tranquil universal victory wisdom excellence youthful zestful",
            "neutral_filler": "word text string letter symbol token element item object thing unit part piece segment portion section component factor aspect feature attribute property characteristic quality trait detail element substance material content information data knowledge fact detail point topic subject matter theme concept idea notion thought principle rule law pattern structure format style method technique approach process procedure system mechanism function operation activity action behavior conduct performance execution implementation application usage utilization employment deployment arrangement organization configuration setup establishment creation formation development construction building assembly production generation manufacture fabrication composition compilation construction"
        },
        "metadata": {
            "description": "Collection of filler texts for CoT intervention experiments",
            "created": "2024-08-09",
            "version": "1.0"
        }
    }

    filepath = "../data/filler_texts.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(filler_texts_data, f, indent=2, ensure_ascii=False)

    print(f"Created {filepath}")
    print("Available filler texts:")
    for name in filler_texts_data["filler_texts"].keys():
        print(f"  - {name}")


if __name__ == "__main__":
    create_filler_texts_file()