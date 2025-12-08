"""Utilities for labeling topic keyword clusters using LLMs."""
import re
import ast
from ollama import chat  # pylint: disable=import-error
from openai import OpenAI  # pylint: disable=import-error
from ollama import ChatResponse # pylint: disable=import-error
from tqdm import tqdm

def parse_model_response(response_text):
    """
    Parse the model's response to extract labels as a single list.
    Handles responses that have multiple lists separated by newlines.

    Args:
        response_text: Raw response from the model

    Returns:
        A single list containing all labels
    """
    # Remove any leading/trailing whitespace
    response_text = response_text.strip()

    # Try to find all list patterns in the text
    # Match patterns like ['label1', 'label2'] or ["label1", "label2"]
    list_pattern = r'\[([^\]]+)\]'
    matches = re.findall(list_pattern, response_text)

    if matches:
        # Extract all labels from all matched lists
        all_labels = []
        for match in matches:
            # Parse each individual list
            try:
                # Reconstruct the list string and evaluate it
                list_str = f'[{match}]'
                parsed_list = ast.literal_eval(list_str)
                all_labels.extend(parsed_list)
            except (ValueError, SyntaxError):
                items = [item.strip().strip('"\'') for item in match.split(',')]
                all_labels.extend(items)


        return str(all_labels)   
    return response_text

def label_topics(df, model_name, api_key_path = None):
    """
    Label topic keyword clusters using either a local model (via Ollama) or an API-based model.

    Args:
        df: DataFrame with TopicBench keyword clusters to label
        model_name: Name of the model (e.g., 'llama3.2:latest' for local, 'gpt-4' for API)
        api_key_path: Path to API key file (None for local models)

    Returns:
        DataFrame with a new column named after the model containing the labels
    """

    labels = []

    if api_key_path is None:
        # Local model
        for idx in tqdm(range(len(df)), desc=f"Labeling topics with {model_name}"):
            row_data = df.iloc[idx]

            # Extract row-specific data
            field = row_data.get('field', '')
            keywords = row_data.get('keywords', '')

            # Create prompt with row-specific values
            formatted_prompt = (
                f"You are a computational social scientist in the field of {field}.\n\n"
                "You have performed topic modeling and need to label the following "
                "keyword clusters.\n\n"
                "You must provide the best unifying high-level concept as a label "
                "for each keyword cluster.\n\n"
                "YOUR RESPONSE MUST ONLY BE LABELS IN THE FORM\n"
                "['label1', 'label2', 'labeln'] for n topics.\n\n"
                f"Top keywords for each topic: {keywords}"
            )

            response: ChatResponse = chat(model=model_name, messages=[
                {
                    'role': 'user',
                    'content': formatted_prompt,
                },
            ])
            # Parse the response to handle multiple lists separated by newlines
            parsed_response = parse_model_response(response['message']['content'])
            labels.append(parsed_response)

    else:
        # API-based model (e.g., OpenAI)
        with open(api_key_path, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()

        client = OpenAI(api_key=api_key)

        for idx in tqdm(range(len(df)), desc=f"Labeling topics with {model_name}"):
            row_data = df.iloc[idx]

            # Extract row-specific data
            field = row_data.get('field', '')
            keywords = row_data.get('keywords', '')

            # Create prompt with row-specific values
            formatted_prompt = f'''
            You are a computational social scientist in the field of {field}.

            You have performed topic modeling and need to label the following keyword clusters.

            You must provide the best unifying high-level concept as a label for each keyword cluster.

            YOUR RESPONSE MUST ONLY BE LABELS IN THE FORM
            ['label1', 'label2', 'labeln'] for n topics.

            Top keywords for each topic: {keywords}
            '''

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': formatted_prompt,
                    }
                ]
            )
            # Parse the response to handle multiple lists separated by newlines
            parsed_response = parse_model_response(response.choices[0].message.content)
            labels.append(parsed_response)

    # Add labels as a new column with the model name
    df[model_name] = labels
    return df
