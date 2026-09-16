"""PrefEval prompt helpers, adapted for OpenAI-compatible chat messages.

Derived from amazon-science/PrefEval; CC-BY-NC-4.0 (see LICENSE).
Unused provider formats and experiment utilities are omitted.
"""

def create_user_pref_message(preference, model_type, system_prompt):
    user_message = [{'role': 'user', 'content': preference}]
    return user_message

def get_question_prompt(preference, pref_generation, question, multi_inter_message, model_type, turn_number, remind, cot, args, max_tokens, system_prompt='You are a helpful assistant.'):
    question += f' (Please respond within {max_tokens} words.)'
    user_message = {'role': 'user', 'content': preference}
    messages = [user_message, {'role': 'assistant', 'content': pref_generation}]
    if multi_inter_message:
        assert turn_number > 0
        messages.extend(multi_inter_message)
    messages.append({'role': 'user', 'content': question})
    return messages
