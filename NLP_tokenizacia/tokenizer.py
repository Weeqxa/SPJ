import re



sentence = ""
sentences = []

original_sentence = []
original_sentences = []

skip_until = 0


with open("sk_snk-ud-train.conllu", "r", encoding="utf-8") as file:
    for line in file:

        if line.startswith("#"):
            continue

        if line.strip() == "":
            sentences.append(sentence)
            original_sentences.append(original_sentence)

            sentence = ""
            original_sentence = []

            skip_until = 0
            continue

        parts = line.strip().split("\t")

        token_id = parts[0]

        form = parts[1]
        misc = parts[9]

        if "-" in token_id:
            start, end = token_id.split("-")
            skip_until = int(end)

            original_sentence.append(form)

        elif token_id.isdigit() and int(token_id) <= skip_until:
            continue

        else:
            original_sentence.append(form)

        if "SpaceAfter=No" in misc:
            sentence += form

        else:
            sentence += form + " "  


correct_sentences = 0
total_sentences = len(sentences)

total_original_tokens = 0
total_custom_tokens = 0

matching_tokens = 0

for i in range(len(sentences)):
    
    test = sentences[i]
    
    tokens = re.findall(r"\w+|--|[^\w\s]", test, re.UNICODE)

    if tokens == original_sentences[i]:
        correct_sentences += 1

        total_original_tokens += len(original_sentences[i])
        total_custom_tokens += len(tokens)

    else:
        print("DIFFERENT")
        print("Original:", original_sentences[i])
        print("Custom:  ", tokens)


    for original, custom in zip(original_sentences[i], tokens):
        if original == custom:
            matching_tokens += 1


accuracy = (matching_tokens / total_original_tokens) * 100 

print(f"Correctly recognized tokens: {matching_tokens}")
print(f"Incorrectly recognized tokens: {total_original_tokens - matching_tokens}")
print(f"Accuracy: {accuracy:.2f}%")





