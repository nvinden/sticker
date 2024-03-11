import re

# Utilities: Tools that help with NLP but are not POS taggers

def clean_text(text):
    # Step 1: Convert to lowercase
    lower_text = text.lower()
    
    # Step 2: Remove everything except lowercase letters, numbers, and spaces
    # This regex replaces anything that is not a lowercase letter, number, or space with nothing
    cleaned_text = re.sub(r'[^a-z0-9 ]+', '', lower_text)
    
    return cleaned_text



def get_sub_phrases(chunk):
    """
    Procedurally extracts sub-phrases from a chunk if it contains more than one word,
    with a maximum of 50 sub-phrases.
    """
    sub_phrases = set()
    words = chunk.split()
    
    # If the chunk is a single word, just return it
    if len(words) <= 1:
        return {chunk}
    
    # Initialize a queue with the original chunk
    queue = [words]
    
    while queue and len(sub_phrases) < 50:
        current_words = queue.pop(0)
        for i in range(len(current_words)):
            # Exclude the current word to form a sub-phrase
            sub_phrase = ' '.join(current_words[:i] + current_words[i+1:])
            if sub_phrase and sub_phrase not in sub_phrases:
                sub_phrases.add(sub_phrase)
                queue.append(sub_phrase.split())
                
    return sub_phrases

def extract_phrases(text, nlp):
    """
    Extracts phrases and sub-phrases from the given text, procedurally.
    Limits the number of sub-phrases to a maximum of 50 for each original phrase.
    """
    # Clean up the text
    text = clean_text(text)

    doc = nlp(text)
    phrases = set(chunk.text for chunk in doc.noun_chunks)  # Initial noun chunks
    
    all_phrases = set()
    for phrase in phrases:
        # Add the phrase itself
        all_phrases.add(phrase)
        # Procedurally extract and add sub-phrases
        sub_phrases = get_sub_phrases(phrase)
        all_phrases.update(sub_phrases)
    
    # Remove all one-letter phrases
    all_phrases = set([p.replace(".", "") for p in all_phrases if len(p) > 1])
    
    return all_phrases


def extract_phrases_hierarchical(text, nlp):
    """
    Extracts phrases and sub-phrases from the given text, procedurally.
    Limits the number of sub-phrases to a maximum of 50 for each original phrase.
    """
    # Clean up the text
    text = clean_text(text)

    doc = nlp(text)

    # Get the valid subsentences
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    # Valid subsequences are noun phrases, nouns, and the original text
    valid_subsentences = noun_phrases + nouns + [text]

    return tuple(valid_subsentences)



def _find_noun_branches(token, branch=None):
    if branch is None:
        branch = []
    branch.append(token)
    
    # Check if the current token is a noun
    has_noun = token.pos_ in ["NOUN", "PROPN"]
    
    # Recursively search for nouns in the children
    for child in token.children:
        temp_branch, temp_has_noun = _find_noun_branches(child, list(branch))
        if temp_has_noun:
            has_noun = True
            print("Branch with noun:", " -> ".join([t.text for t in temp_branch]))
    
    return branch, has_noun