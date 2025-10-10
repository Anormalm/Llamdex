from transformers import AutoTokenizer

def print_alphabet_token_ids(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    alphabet = [chr(i) for i in range(65, 91)]

    print(f"Token IDs for alphabet letters in {model_name}:")

    token_id_list = []

    for letter in alphabet:
        token_id = tokenizer.encode(letter, add_special_tokens=False)[0]
        token_id_list.append(token_id)

    print(token_id_list)

if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print_alphabet_token_ids(model_name)