"""PrefEval prompt helpers, adapted for OpenAI-compatible chat messages.

Derived from amazon-science/PrefEval; CC-BY-NC-4.0 (see LICENSE).
Unused provider formats and experiment utilities are omitted.
"""

def extract_multi_turn_conversation(multi_turn_message, turn_number=3, model_type='claude'):
    """Render the mid-conversation filler turns into `model_type`'s message format.

    `model_type` is kept for compatibility. All OpenAI-compatible backends use
    the "claude" branch (a list of role/content dicts).
    """
    message = []
    for turn in multi_turn_message:
        role = turn['role']
        content = turn['content']
        message.append({'role': role, 'content': content})
        if len(message) == turn_number * 2:
            if role != 'assistant':
                raise ValueError('The last turn must be from assistant')
            break
    assert len(message) == turn_number * 2, 'The number of turns is less than the specified number'
    return message
ALL_TOPICS = ['travel_transportation', 'shop_motors', 'lifestyle_beauty', 'travel_restaurant', 'shop_fashion', 'entertain_shows', 'pet_ownership', 'lifestyle_fit', 'entertain_games', 'shop_home', 'lifestyle_health', 'travel_activities', 'education_learning_styles', 'entertain_music_book', 'professional_work_location_style', 'education_resources', 'lifestyle_dietary', 'shop_technology', 'travel_hotel', 'entertain_sports']
