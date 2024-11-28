import json

input_file = '/home/IAIS/jdatta/distillm-new/data/squad_json/dev.jsonl'
output_file = '/home/IAIS/jdatta/distillm-new/data/squad_json/valid.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        # Extract required fields
        id = data.get('id', '')
        context = data.get('context', '')
        question = data.get('question', '')
        # Build the prompt
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        # Determine the output
        answers = data.get('answers', {}).get('text', [])
        if answers:
            output = answers[0]
        else:
            output = "No answer available."
        # Create the new JSON object
        new_data = {
            'id': id,
            'prompt': prompt,
            'output': output
        }
        # Write to the output file
        json.dump(new_data, outfile)
        outfile.write('\n')
