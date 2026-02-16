InfoHub RAG Assistant

ეს არის Retrieval-Augmented Generation (RAG) სისტემა, რომელიც პასუხობს ქართულად დასმულ კითხვებს „საინფორმაციო და მეთოდოლოგიური ჰაბის“ დოკუმენტების საფუძველზე.

სისტემა იყენებს:

OpenAI Embeddings (text-embedding-3-large)

GPT-4o (პასუხის გენერაცია)

Qdrant (vector database)

FastAPI (API layer)

პასუხი ყოველთვის ეფუძნება მოძიებულ კონტექსტს და შეიცავს წყაროს მითითებას.

მოთხოვნები

სისტემის გასაშვებად საჭიროა:

Python 3.11+

Docker Desktop

OpenAI API Key

data/infohub.csv (UTF-8 ფორმატში)

1. Qdrant-ის გაშვება

Docker Desktop უნდა იყოს ჩართული.

ტერმინალში გაუშვით:

docker pull qdrant/qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant


შემოწმება:

გახსენით ბრაუზერში:

http://localhost:6333


თუ JSON პასუხი დაბრუნდა — Qdrant მუშაობს.

2. გარემოს კონფიგურაცია

პროექტის root-ში შექმენით .env ფაილი:

OPENAI_API_KEY=your_api_key_here

3. დამოკიდებულებების ინსტალაცია

პროექტის root-ში:

pip install -r requirements.txt

4. მონაცემთა ინდექსაცია (Ingestion)

CSV ფაილი უნდა იყოს:

data/infohub.csv


შემდეგ გაუშვით:

python -m app.ingestion


ინდექსაციის დასრულების შემდეგ შეგიძლიათ გადაამოწმოთ:

http://localhost:6333/collections/infohub_documents


points_count უნდა იყოს 0-ზე მეტი.

5. API გაშვება
python -m uvicorn app.main:app --reload


შემდეგ გახსენით:

http://127.0.0.1:8000/docs

6. ტესტირება

POST /ask endpoint-ზე გაგზავნეთ:

{
  "question": "რა არის იმპორტის დეკლარაცია?"
}


სისტემა დააბრუნებს:

პასუხს ქართულად

წყაროს ბლოკს

confidence_score-ს

უსაფრთხოების მექანიზმი

თუ ინფორმაცია მოძიებულ დოკუმენტებში არ არსებობს, სისტემა აბრუნებს:

მოცემულ დოკუმენტებში ინფორმაცია ვერ მოიძებნა.


სისტემა არ იგონებს პასუხებს და არ ქმნის არარსებულ სამართლებრივ მითითებებს.

არქიტექტურა
User Question
      ↓
Embedding
      ↓
Qdrant Vector Search (top_k=8)
      ↓
Context Assembly
      ↓
GPT-4o Generation
      ↓
Structured Georgian Answer + Source

გამოყენებული ტექნოლოგიები

FastAPI

OpenAI API

Qdrant

Pandas

Tiktoken

Docker

დამატებითი შესაძლებლობები

Conversation memory (ბოლო 5 კითხვა-პასუხი)

Confidence scoring (similarity average)

Hallucination protection

სრული მეტამონაცემების შენახვა Qdrant-ში

თუ ingestion დასრულდა და API გაშვებულია, სისტემა მზად არის გამოყენებისთვის.