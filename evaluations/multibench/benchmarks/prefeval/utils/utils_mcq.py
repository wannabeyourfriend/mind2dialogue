"""PrefEval prompt helpers, adapted for OpenAI-compatible chat messages.

Derived from amazon-science/PrefEval; CC-BY-NC-4.0 (see LICENSE).
Unused provider formats and experiment utilities are omitted.
"""
import random
import re
from bs4 import BeautifulSoup

def get_mcq_question_format(options):
    """
    Generate the question format with provided options.

    :param options: List of options to be included in the question format
    :return: A formatted MCQ question string
    """
    formatted_options = format_options(options)
    mcq_question_format = f"""\n    I'm trying to decide on this and here are 4 options for my query: \n{formatted_options}\nNow, I'd like you to pick one of them as your top recommendation for me.\n    Important instructions for your response:\n    1. Choose only one option (A, B, C, or D) that best matches my preferences.\n    2. Your answer must be one of these options.\n    3. Don't say things like "I can't choose" or suggest alternatives not listed.\n    4. Answer example: <choice>B</choice>. Give me your answer in this exact format, without any additional explanation:\n       <choice>[A/B/C/D]</choice>\n    """
    return mcq_question_format

def extract_choice(response):
    """
    Extract the choice (A, B, C, or D) from the LLM's response.

    :param response: String containing the LLM's response
    :return: The extracted choice or None if no valid choice is found
    """
    soup = BeautifulSoup(response, 'html.parser')
    choice_tag = soup.find('choice')
    try:
        if choice_tag:
            choice_content = choice_tag.string
            match = re.search('[ABCD]', choice_content)
            if match:
                return match.group(0)
    except:
        return None

def shuffle_options(options):
    """
    Note: In the MCQ datasets, the first choice in the JSON file is the correct answer.
    This function shuffles the options randomly while keeping track of the correct answer.

    :param options: List of options where the first option is the correct answer
    :return: Tuple containing the shuffled options and the index of the correct answer
    """
    correct_answer = options[0]
    shuffled_options = random.sample(options, len(options))
    correct_index = shuffled_options.index(correct_answer)
    return (shuffled_options, correct_index)

def format_options(options):
    """
    Format a list of options into a lettered string.

    :param options: List of option strings
    :return: Formatted string with lettered options
    """
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    formatted_options = []
    for i, option in enumerate(options):
        formatted_options.append(f'{letters[i]}. {option}')
    return '\n'.join(formatted_options)

def get_question_prompt_mcq(preference, options, pref_generation, question, multi_inter_message, model_type, turn_number, remind, cot, system_prompt='You are a helpful assistant.'):
    mcq_question_format = get_mcq_question_format(options)
    user_message = {'role': 'user', 'content': preference}
    messages = [user_message, {'role': 'assistant', 'content': pref_generation}]
    if multi_inter_message:
        assert turn_number > 0
        messages.extend(multi_inter_message)
    messages.append({'role': 'user', 'content': question + mcq_question_format})
    messages.append({'role': 'assistant', 'content': '<choice>'})
    return messages
