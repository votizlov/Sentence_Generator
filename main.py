import sys

import numpy as np
import random
import re
from collections import defaultdict
import codecs
import markovify
import json


def generate(mode):
    path = '1.txt'
    text = codecs.open(path, "r", "utf_8_sig").read()
    if mode == "normal":
        # Read text from file and tokenize.

        # Build the model.
        text_model = markovify.Text(text)

        # Print five randomly-generated sentences
        return text_model.make_sentence()

        # Print three randomly-generated sentences of no more than 280 characters
        # for i in range(3):
        #    print(text_model.make_short_sentence(280))
    elif mode == "extreme":
        # Create graph.
        markov_graph = defaultdict(lambda: defaultdict(int))
        tokenized_text = [
            word
            for word in re.split('\W+', text)
            if word != ''
        ]
        last_word = tokenized_text[0].lower()
        for word in tokenized_text[1:]:
            word = word.lower()
            markov_graph[last_word][word] += 1
            last_word = word

        def walk_graph(graph, distance=5, start_node=None):
            """Returns a list of words from a randomly weighted walk."""
            if distance <= 0:
                return []

            # If not given, pick a start node at random.
            if not start_node:
                start_node = random.choice(list(graph.keys()))

            weights = np.array(
                list(markov_graph[start_node].values()),
                dtype=np.float64)
            # Normalize word counts to sum to 1.
            weights /= weights.sum()

            # Pick a destination using weighted distribution.
            choices = list(markov_graph[start_node].keys())
            chosen_word = np.random.choice(choices, None, p=weights)

            return [chosen_word] + walk_graph(
                graph, distance=distance - 1,
                start_node=chosen_word)

        s = ' '
        for i in range(10):
            s += ' '.join(walk_graph(
                markov_graph, 10)), '\n'
            return json.dumps(s)


print(generate("normal")).encode("utf-8")
