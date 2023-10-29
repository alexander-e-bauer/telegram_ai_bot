# Imports
import os
import requests
import re
import openai
import tiktoken
import pandas as pd
from scipy import spatial
from bs4 import BeautifulSoup
from googleapiclient.discovery import build


# Define Configuaration Constants
"""Requires Google Custom Search API key and search engine ID, and OpenAI API key."""
google_search_key = os.getenv("GOOGLE_SEARCH_KEY")
google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use
MAX_TOKENS = 1600
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request
error = "My search didn't find any relevant results, but try asking again with a slightly different format."

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(str(text)))



def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    for i in string:
        chunks = i.split(delimiter)
        if len(chunks) == 1:
            return [string, ""]  # no delimiter found
        elif len(chunks) == 2:
            return chunks  # no need to search for halfway point
        else:
            total_tokens = num_tokens(string)
            halfway = total_tokens // 2
            best_diff = halfway
            for i, chunk in enumerate(chunks):
                left = delimiter.join(chunks[: i + 1])
                left_tokens = num_tokens(left)
                diff = abs(halfway - left_tokens)
                if diff >= best_diff:
                    break
                else:
                    best_diff = diff
            left = delimiter.join(chunks[:i])
            right = delimiter.join(chunks[i:])
            return [left, right]


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(str(string))
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


def split_strings_from_subsection(
    subsection,
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    print(subsection)
    string = subsection
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(string, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_subsection = (subsection, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles to answer the subsequent question. ' \
                   'If the answer cannot be found in the articles, try to answer the question '
    question = f"\n\nQuestion: {query}"
    message = introduction
    return message + question


def ask(
    query: str,
    df: pd.DataFrame,
    location_id: str,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 100,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": f"You answer questions about plants, while attempting to provide "
                                      f"results specific to {location_id}. "
                                      f"Include interesting scientific facts about plants if appropriate. "
                                      f"if a specific plant is mentioned, provide an interesting fact about the plant "
                                      f"and provide a list of other plants that like to be companion plants with it."
                                      f"provide a link to more information about the plant or concept at the end."},

        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

def google(query, number, location_id):
    search_strings = []
    embeddings = []
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.
    service = build(
        "customsearch", "v1", developerKey=google_search_key
    )

    res = (
        service.cse()
        .list(
            q=query,
            cx=google_search_engine_id,
            num=number,
        )
        .execute()
    )
    print(res)
    output, snippets = [], ""
    items = res["items"]

    for item in items:
        title = item["title"]
        snippet = item["snippet"]
        snippets = snippets + snippet

        link = item["link"]
        response = requests.get(link)
        html = response.content

        # Assume 'html' is the HTML code you want to filter
        soup = BeautifulSoup(html, 'html.parser')

        # Use find_all() to find elements with text matching a specific pattern using regular expressions
        pattern = re.compile(r'\b[A-Za-z]+(?:\s[A-Za-z]+)*[.,]\s(?:\b[A-Za-z]+(?:\s[A-Za-z]+)*[.,]\s?)*')
        filtered_elements = soup.find_all(string=pattern)

        # Extract the text from the filtered elements
        text = [text.get_text() for text in filtered_elements]
        for i in text:

            i = i.replace('\n', '')
            i = i.replace('\t', '')
            i = i.replace('\r', '')
            #print(i)
            output.append(i)

    # split sections into chunks
    search_strings.extend(split_strings_from_subsection(output, max_tokens=MAX_TOKENS))

    print(f"{len(output)} Web Page sections split into {len(search_strings)} strings.")

    for batch_start in range(0, len(search_strings), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = search_strings[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end - 1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)

    print(f"{len(embeddings)} embeddings retrieved.")

    if len(search_strings) != len(embeddings):
        # Handle error here, such as logging or raising an exception
        print("Error: number of strings and embeddings do not match.")
        return error
    else:
        df = pd.DataFrame({"text": search_strings, "embedding": embeddings})

    try:
        results = ask(query, df, location_id)
        df = []
    except:
        results = error
        pass

    return results
