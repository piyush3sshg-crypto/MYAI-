"""Built-in toy datasets for demos, tests, and experiment notebooks.

These wrap the same example sentences main.py seeds into the chatbot's
intent classifier, so the data science layer has something real to run
descriptive stats / cross-validation against without needing an external
data file.
"""
from data.dataset import Dataset


INTENT_EXAMPLES = {
    "greeting": [
        "hi there", "good morning", "hey how are you", "namaste",
        "hello", "yo whats up", "good evening", "hii",
        "hey", "good afternoon", "hi how is it going", "whats up",
        "morning", "hello there friend", "greetings",
        "hey buddy", "hi again", "namaskar",
        "hiya", "good to see you", "hello friend how are things",
        "hey there stranger", "sup", "hi good morning to you",
        "hello good day", "hey hows life", "namaste ji",
        "hi hi", "hello there how are you doing", "yo",
        "good day to you",
    ],
    "farewell": [
        "goodbye", "see you later", "bye take care",
        "catch you later", "i am leaving now", "farewell then",
        "bye bye", "see ya", "gotta go now", "talk to you soon",
        "signing off", "i am off now", "later then",
        "take care bye", "alvida",
        "see you tomorrow", "gotta run bye", "catch ya later",
        "im heading out now", "bye for now", "take care of yourself",
        "see you soon then", "okay im leaving", "bye have a good one",
        "i must go now goodbye", "talk soon bye", "peace out",
        "im out bye", "alright bye then", "ok bye see you",
    ],
    "math_query": [
        "what is 12 plus 45",
        "solve 2x + 4 = 10",
        "calculate the square root of 81",
        "evaluate 5 times 7 minus 3",
        "what is 9 divided by 3",
        "what is 100 minus 37",
        "add 23 and 56",
        "what is 8 times 9",
        "solve for x in 3x = 15",
        "whats 200 divided by 4",
        "subtract 15 from 90",
        "what is the value of 7 squared",
        "compute 45 plus 78",
        "what is 6 multiplied by 6",
        "solve 5x - 2 = 18",
        "can you calculate 34 plus 21",
        "please solve this equation 4x = 20",
        "what does 15 times 15 equal",
        "find the sum of 12 and 88",
        "divide 144 by 12 for me",
        "what is the square root of 144",
        "multiply 7 by 8 please",
        "what is 500 minus 250",
        "solve the equation 6x + 3 = 27",
        "calculate 9 squared",
        "what is the product of 4 and 25",
        "add up 17 and 33 for me",
        "what number is 3 times 3 times 3",
        "please compute 81 divided by 9",
        "solve 10x = 100 for x",
    ],
    "family_query": [
        "who is the father of vihaan",
        "who are the grandparents of kabir",
        "is arjun an ancestor of ishaan",
        "tell me about the family tree",
        "who is vihaan related to",
        "who is kabir's father",
        "list all the grandparents",
        "who are ishaan's ancestors",
        "is meera related to kabir",
        "who is vihaan's son",
        "tell me the family relations",
        "who is arjun's grandchild",
        "who is the grandmother in this family",
        "show me the family relationships",
        "who are arjun's descendants",
        "is vihaan kabir's parent",
        "who is meera's grandson",
        "explain the family connections here",
        "who is related to arjun as a child",
        "who is the eldest in the family tree",
        "who is ishaan's grandfather",
        "list the parents in this family",
        "who are all the children of arjun",
        "how is meera connected to ishaan",
        "who is vihaan's father",
    ],
    "weather_query": [
        "mausam kaisa hai",
        "kya barish ho rahi hai",
        "aaj dhoop hai kya",
        "how is the weather today",
        "is it raining outside",
        "will it be sunny tomorrow",
        "what is the temperature right now",
        "is it going to be cold today",
        "kal mausam kaisa rahega",
        "how humid is it outside",
        "is there a storm coming",
        "whats the forecast for today",
        "will it snow this week",
        "kya aaj thand hai",
        "what is the weather forecast for tomorrow",
        "how windy is it today",
        "is the sky clear right now",
        "will it be hot this afternoon",
        "kya mausam saaf hai",
        "what is today's humidity level",
        "is a storm expected this weekend",
        "how cloudy is it outside",
        "will there be rain tonight",
        "kitni garmi hai aaj",
        "is it going to be windy tomorrow",
    ],
}


def load_intent_dataset(version="v1"):
    """Return the intent-classification examples as a Dataset of
    {"text": ..., "label": ...} records — the shape most ML utilities
    (metrics, cross-validation) expect.
    """
    records = []
    for label, examples in INTENT_EXAMPLES.items():
        for text in examples:
            records.append({"text": text, "label": label})
    return Dataset(records, name="intent_examples", version=version)


def load_spam_dataset(version="v1"):
    """Load the ham/spam CSV dataset from data/spam_dataset.csv.

    Honest disclosure: this is a template-generated synthetic dataset
    (160 rows, ~90 ham / ~70 spam), not scraped from a real corpus —
    there's no network access to fetch one. It exists to demonstrate
    the *pipeline* (CSV load -> EDA -> features -> CV -> report) on a
    binary classification problem with a meaningfully larger row count
    and two well-separated classes, which is what the intent dataset
    (31-100 rows, 5 overlapping classes) can't show on its own.
    """
    import os
    path = os.path.join(os.path.dirname(__file__), "spam_dataset.csv")
    return Dataset.from_csv(path, name="spam_dataset", version=version)
