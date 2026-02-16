NOT_FOUND_MESSAGE = "მოცემულ დოკუმენტებში ინფორმაცია ვერ მოიძებნა."


SYSTEM_PROMPT = """
თქვენ ხართ ოფიციალური საგადასახადო და საბაჟო სამართლებრივი ასისტენტი.

წესები:

1. უპასუხეთ მხოლოდ მიწოდებული კონტექსტის საფუძველზე.
2. არ გამოიგონოთ ინფორმაცია.
3. არ მიუთითოთ არარსებული სამართლებრივი ნორმები.
4. თუ ინფორმაცია ნაწილობრივ მოიძებნა — გააკეთეთ ზუსტი, პროფესიული შეჯამება.
5. თუ პასუხი საერთოდ არ არსებობს კონტექსტში — უპასუხეთ ზუსტად:

მოცემულ დოკუმენტებში ინფორმაცია ვერ მოიძებნა.

პასუხი უნდა იყოს:
- პროფესიული
- ფორმალური
- ბუნებრივი ქართული ენით
- არა რობოტული
- სტრუქტურირებული აბზაცებად

პასუხის ბოლოს ყოველთვის დაამატეთ წყაროს ბლოკი მოცემული ფორმატით.
არ გაიმეოროთ ტექსტები.
"""


def _clean_value(value: str) -> str:
    """Removes duplicated prefixes like 'კატეგორია: კატეგორია:'"""
    if not value:
        return ""
    value = value.strip()
    value = value.replace("კატეგორია: ", "")
    value = value.replace("მიღების თარიღი: ", "")
    value = value.replace("ბრძანება N ", "")
    return value.strip()


def build_context_message(context_chunks: list[dict]) -> str:
    blocks: list[str] = []

    for index, chunk in enumerate(context_chunks, start=1):
        metadata = chunk.get("metadata", {})

        document = _clean_value(metadata.get("დოკუმენტი", ""))
        order = _clean_value(metadata.get("ბრძანება", ""))
        category = _clean_value(metadata.get("კატეგორია", ""))
        date = _clean_value(metadata.get("მიღების თარიღი", ""))
        link = metadata.get("item_page_link", "")

        blocks.append(
            "\n".join(
                [
                    f"[კონტექსტი {index}]",
                    f"მსგავსების ქულა: {chunk.get('score', 0.0):.4f}",
                    f"დოკუმენტი: {document}",
                    f"ბრძანება: {order}",
                    f"კატეგორია: {category}",
                    f"მიღების თარიღი: {date}",
                    f"ლინკი: {link}",
                    "ტექსტი:",
                    chunk.get("text", ""),
                ]
            )
        )

    return (
        "ქვემოთ მოცემულია მოძიებული სამართლებრივი კონტექსტი.\n"
        "გამოიყენეთ მხოლოდ ეს ინფორმაცია პასუხის ფორმირებისთვის.\n\n"
        + "\n\n".join(blocks)
    )


def build_latest_question_message(question: str) -> str:
    return (
        "უპასუხე ქვემოთ მოცემულ შეკითხვას მხოლოდ კონტექსტზე დაყრდნობით.\n"
        "პასუხი უნდა იყოს პროფესიული, სამართლებრივი და სტრუქტურირებული.\n"
        "თუ პასუხი არ მოიძებნა, გამოიყენე ზუსტი ფორმულირება:\n"
        f"{NOT_FOUND_MESSAGE}\n\n"
        f"შეკითხვა:\n{question}"
    )


def format_source_block(sources: list[dict]) -> str:
    lines = ["წყარო:"]

    if not sources:
        lines.append("არ მოიძებნა")
        return "\n".join(lines)

    seen = set()

    for source in sources:
        document = _clean_value(source.get("document", ""))
        order = _clean_value(source.get("order_number", ""))
        category = _clean_value(source.get("category", ""))
        date = _clean_value(source.get("date", ""))
        url = source.get("url", "")

        unique_key = (document, order, url)
        if unique_key in seen:
            continue
        seen.add(unique_key)

        lines.extend(
            [
                f"{document} – {order}",
                f"კატეგორია: {category}",
                f"მიღების თარიღი: {date}",
                f"ლინკი: {url}",
                "",
            ]
        )

    return "\n".join(lines).strip()
