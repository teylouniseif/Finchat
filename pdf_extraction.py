from transformers import pipeline
import tabula, openai, pdfplumber, os, math, json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_KEY")

def chunk_text(text, max_chars=512):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def extract_text_from_pdf(pdf_path):
    result = {}
    with pdfplumber.open(pdf_path) as pdf:
        pages, chunks_text = [], []
        for page_num, page in enumerate(pdf.pages, start=1):
            pages.append(page)
            chunks_text += chunk_text(page.extract_text())
        #extract tables from pages
        tables = []
        for page_num, page in enumerate(pages):
            tables_data = tabula.read_pdf(pdf_path, pages=page_num+1, multiple_tables=True, output_format='json')
            df_tables_data = tabula.read_pdf(pdf_path, pages=page_num+1, multiple_tables=True)
            for id, table in enumerate(tables_data):
                table_top = table['top'] if 'top' in table else 0
                text_near_table = ''.join(page.within_bbox((0,0, page.width, table_top)).extract_text().splitlines()[-15:])
                tables.append(
                    text_near_table + ' Table: ' + str(df_tables_data[id].to_dict(orient='records'))
                )
        result['tables'] = tables
        result['text'] = chunks_text

    f = open('filing.json', 'w')
    json.dump(result, f)


# Read PDF, convert to text
pdf_path = "filing.pdf"  # Update this path to your PDF file
text_pages = extract_text_from_pdf(pdf_path)
    
with open('filing.json', 'r') as file:
    text_pages = json.load(file)

# Load a BERT-based question-answering pipeline
qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Define the question to ask
#question = "What is the total revenue of the company for 2023?"
#question = "What is the gross profit of the company for 2022?"
question = 'What is the percentage of revenue of Asia Pacific region for 2021?'

#Answer the question for each chunk and store answers with probabilities
answers = []
#looking at tables only here
for i, chunk in enumerate(text_pages['tables']):# + text_pages['text']):
    result = qa_model(question=question, context=str(chunk))
    answers.append((result['answer'], result['score'], str(chunk)))
    print(f"Chunk {i+1} Answer: {result['answer']}")
    print(f"Probability: {result['score']}")

#can also place a threshold for score minimum if desired, for now just taking top scores
answers = sorted(answers, key=lambda x: x[1], reverse=True)[:10]


# use LLM to decide which of answers is the correct one.
context = f"""
            Considering this list of tuples, with first element being potential answer to question from QA model, second score
            of potential answer, and third text from which answer is taken, what is the answer to the question: {question}
            List of tuples is {answers}
"""

openai.api_key = api_key
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a financial analyst assistant."},
        {"role": "user", "content": f"{question}: {context}"}
    ],
    temperature=0.2,
    logprobs=True
)

answer = response.choices[0].message.content
tokens_logprobs = response.choices[0].logprobs.content

# Average probability of answer
token_probs = [math.exp(el.logprob) for el in tokens_logprobs]
average_prob = sum(token_probs) / len(token_probs) if token_probs else None


# if probabilty is below threshold, raise flag for manual verification, 90% seems like good minimal threshold
print(f'Answer: {answer}\n Confidence: {average_prob}')

#First look through tables, if answer score too low look through text chunks, if still low raise flag
