"""Build the Task 5 gold evaluation set from the frozen Constitution corpus.

Design rules (Task 5 / Docks/README.md section 9):
* Every answerable row references real article(s) and real chunk ID(s) from
  ``Data/processed/constitution_chunks.jsonl``.
* Evidence chunk IDs are resolved by locating the article heading text in the
  corpus, never guessed.
* Unanswerable rows have empty article/evidence lists and say so explicitly.
* Rows are marked ``draft_pending_human_review`` until a human verifies the
  reference answer and evidence labels against the official text.

Run:  python build_test_questions.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

CHUNKS_PATH = Path("Data/processed/constitution_chunks.jsonl")
OUTPUT_PATH = Path("Data/processed/test_questions.jsonl")

DEV_FRACTION = 0.25

CATEGORY_ORDER = [
    "direct_article_question",
    "paraphrased_question",
    "multi_article_question",
    "keyword_heavy_question",
    "ambiguous_question",
    "unanswerable_out_of_corpus",
]

# article, reference answer, topic, keyword, evidence heading
FACTS = [
    ("1", "India, that is Bharat, shall be a Union of States; its territory comprises the territories of the States, the Union territories, and such other territories as may be acquired.", "the name and territory of the Union", "India, that is Bharat", "1. Name and territory of the Union"),
    ("3", "Parliament may by law form a new State, increase or diminish the area of a State, alter its boundaries, or alter its name.", "the formation of new States and alteration of State boundaries", "formation of new States", "3. Formation of new States"),
    ("5", "At the commencement of the Constitution, a person domiciled in India is a citizen if born in India, if either parent was born in India, or if ordinarily resident in India for at least five years before commencement.", "citizenship at the commencement of the Constitution", "five years residence citizenship", "5. Citizenship at the commencement"),
    ("11", "Parliament may make laws on the acquisition and termination of citizenship and all other matters relating to citizenship.", "Parliament's power over citizenship", "acquisition and termination of citizenship", "11. Parliament to regulate the right of citizenship"),
    ("12", "For Part III, 'the State' includes the Government and Parliament of India, State governments and legislatures, and all local or other authorities within India or under the control of the Government of India.", "the definition of 'the State' for fundamental rights", "'the State' definition", "12. Definition"),
    ("13", "Laws in force that are inconsistent with the fundamental rights are void to the extent of the inconsistency, and the State cannot make laws that take away or abridge those rights.", "laws inconsistent with fundamental rights", "laws inconsistent with fundamental rights", "13. Laws inconsistent with or in derogation"),
    ("14", "The State shall not deny to any person equality before the law or the equal protection of the laws within India.", "equality before the law", "equal protection of the laws", "14. Equality before law"),
    ("15", "The State shall not discriminate against any citizen on grounds only of religion, race, caste, sex, or place of birth.", "prohibition of discrimination", "discrimination on grounds of religion, race, caste, sex or place of birth", "15. Prohibition of discrimination"),
    ("16", "There shall be equality of opportunity for all citizens in public employment, and no citizen shall be discriminated against on grounds only of religion, race, caste, sex, descent, place of birth, or residence.", "equality of opportunity in public employment", "public employment equality of opportunity", "16. Equality of opportunity"),
    ("17", "Untouchability is abolished and its practice in any form is forbidden; enforcing any disability arising out of untouchability is an offence punishable by law.", "the abolition of untouchability", "untouchability abolished", "17. Abolition of Untouchability"),
    ("18", "The State shall not confer titles other than military or academic distinctions, and citizens shall not accept titles from foreign States.", "the abolition of titles", "titles military or academic distinction", "18. Abolition of titles"),
    ("19", "All citizens have the right to freedom of speech and expression, to assemble peaceably without arms, to form associations or unions, and to move freely throughout India.", "freedom of speech and other freedoms", "freedom of speech and expression", "19. Protection of certain rights regarding freedom of speech"),
    ("20", "No person shall be convicted of an offence except for violating a law in force when the act was committed, and no penalty greater than that then applicable may be imposed.", "protection in respect of conviction for offences", "conviction for offences penalty", "20. Protection in respect of conviction"),
    ("21", "No person shall be deprived of life or personal liberty except according to procedure established by law.", "the protection of life and personal liberty", "life and personal liberty", "21. Protection of life and personal liberty"),
    ("21A", "The State shall provide free and compulsory education to all children aged six to fourteen years in the manner determined by law.", "the right to education", "free and compulsory education six to fourteen", "21A. Right to education"),
    ("22", "A person who is arrested must be informed of the grounds of arrest and must not be denied the right to consult and be defended by a legal practitioner of their choice.", "protection against arrest and detention", "grounds of arrest legal practitioner", "22. Protection against arrest and detention"),
    ("23", "Traffic in human beings, begar, and other similar forms of forced labour are prohibited, though the State may impose compulsory service for public purposes without discrimination.", "the prohibition of traffic in human beings and forced labour", "traffic in human beings begar forced labour", "23. Prohibition of traffic in human beings"),
    ("24", "No child below the age of fourteen years shall be employed in any factory or mine or engaged in any other hazardous employment.", "the prohibition of child employment", "child below fourteen factory mine hazardous employment", "24. Prohibition of employment of children"),
    ("25", "Subject to public order, morality, and health, all persons are equally entitled to freedom of conscience and the right to profess, practise, and propagate religion.", "freedom of religion", "freedom of conscience profession practice propagation", "25. Freedom of conscience"),
    ("28", "No religious instruction shall be provided in an educational institution wholly maintained out of State funds, and no person may be compelled to take part in religious instruction or attend religious worship.", "religious instruction in educational institutions", "religious instruction State funds", "28. Freedom as to attendance at religious instruction"),
    ("29", "Any section of citizens with a distinct language, script, or culture has the right to conserve it, and no citizen may be denied admission to a State-maintained or State-aided institution on grounds only of religion, race, caste, or language.", "the protection of minority interests", "distinct language script culture", "29. Protection of interests of minorities"),
    ("30", "All minorities, whether based on religion or language, have the right to establish and administer educational institutions of their choice.", "minorities' educational institutions", "minorities establish administer educational institutions", "30. Right of minorities to establish"),
    ("32", "The right to move the Supreme Court for enforcement of fundamental rights is guaranteed, and the Court may issue writs including habeas corpus, mandamus, prohibition, quo warranto, and certiorari.", "remedies for enforcement of fundamental rights", "habeas corpus mandamus prohibition quo warranto certiorari", "32. Remedies for enforcement of rights"),
    ("36", "In Part IV, 'the State' has the same meaning as in Part III.", "the definition of 'the State' for directive principles", "State same meaning as Part III", "36. Definition"),
    ("38", "The State shall strive to promote the welfare of the people by securing a social order in which justice, social, economic, and political, informs all national institutions, and shall strive to minimise inequalities in income and status.", "the promotion of a social order and welfare", "social order justice social economic political", "38. State to secure a social order"),
    ("39A", "The State shall secure equal justice and provide free legal aid so that no citizen is denied justice because of economic or other disabilities.", "equal justice and free legal aid", "free legal aid economic disabilities", "39A. Equal justice and free legal aid"),
    ("40", "The State shall organise village panchayats and endow them with the powers and authority needed to function as units of self-government.", "village panchayats", "village panchayats units of self-government", "40. Organisation of village panchayats"),
    ("41", "Within the limits of its economic capacity, the State shall make effective provision for the right to work, to education, and to public assistance in cases of unemployment, old age, sickness, and disablement.", "the right to work, education, and public assistance", "right to work education public assistance", "41. Right to work"),
    ("43A", "The State shall take steps to secure the participation of workers in the management of industries.", "workers' participation in management", "participation of workers in management", "43A. Participation of workers in management"),
    ("44", "The State shall endeavour to secure for the citizens a uniform civil code throughout the territory of India.", "a uniform civil code", "uniform civil code", "44. Uniform civil code"),
    ("45", "The State shall endeavour to provide early childhood care and education for all children until they complete the age of six years.", "early childhood care and education", "early childhood care education below six", "45. Provision for early childhood care"),
    ("46", "The State shall promote with special care the educational and economic interests of the weaker sections, especially Scheduled Castes and Scheduled Tribes, and protect them from social injustice and exploitation.", "the interests of Scheduled Castes, Scheduled Tribes, and weaker sections", "Scheduled Castes Scheduled Tribes weaker sections", "46. Promotion of educational and economic interests"),
    ("47", "The State shall regard the raising of the level of nutrition and the standard of living of its people, and the improvement of public health, as among its primary duties.", "nutrition, standard of living, and public health", "level of nutrition standard of living public health", "47. Duty of the State to raise the level of nutrition"),
    ("48A", "The State shall endeavour to protect and improve the environment and to safeguard the forests and wildlife of the country.", "environmental protection", "environment forests wild life", "48A. Protection and improvement of environment"),
    ("49", "The State must protect every monument, place, or object of artistic or historic interest declared to be of national importance from spoliation, disfigurement, destruction, removal, disposal, or export.", "protection of monuments of national importance", "monuments national importance", "49. Protection of monuments"),
    ("50", "The State shall take steps to separate the judiciary from the executive in the public services of the State.", "separation of the judiciary from the executive", "separation of judiciary from executive", "50. Separation of judiciary from executive"),
    ("51A", "Every citizen has fundamental duties, including abiding by the Constitution, respecting the national flag and anthem, protecting sovereignty and integrity, promoting harmony, preserving heritage, protecting the environment, and developing a scientific temper.", "the fundamental duties of citizens", "fundamental duties every citizen", "51A. Fundamental duties"),
    ("52", "There shall be a President of India.", "the office of the President", "President of India", "52. The President of India"),
    ("53", "The executive power of the Union is vested in the President and is exercised directly or through subordinate officers; the supreme command of the Defence Forces is also vested in the President.", "the executive power of the Union", "executive power vested in President", "53. Executive power of the Union"),
    ("54", "The President is elected by an electoral college consisting of the elected members of both Houses of Parliament and the elected members of the Legislative Assemblies of the States.", "the election of the President", "electoral college elected members", "54. Election of President"),
    ("74", "There shall be a Council of Ministers headed by the Prime Minister to aid and advise the President, who shall act in accordance with that advice; the President may require reconsideration.", "the Council of Ministers", "Council of Ministers Prime Minister aid and advise", "74. Council of Ministers to aid and advise"),
    ("76", "The President shall appoint as Attorney-General for India a person qualified to be a Judge of the Supreme Court; the Attorney-General advises the Government of India on legal matters.", "the Attorney-General for India", "Attorney-General legal matters", "76. Attorney-General for India"),
    ("141", "The law declared by the Supreme Court shall be binding on all courts within the territory of India.", "the binding force of Supreme Court decisions", "law declared by Supreme Court binding", "141. Law declared by Supreme Court"),
    ("148", "There shall be a Comptroller and Auditor-General of India, appointed by the President by warrant, who may be removed only in the same manner and on the same grounds as a Judge of the Supreme Court.", "the Comptroller and Auditor-General of India", "Comptroller and Auditor-General", "148. Comptroller and Auditor-General"),
    ("149", "The Comptroller and Auditor-General shall perform the duties and exercise the powers relating to the accounts of the Union and the States as prescribed by law.", "the duties and powers of the Comptroller and Auditor-General", "accounts of the Union and States", "149. Duties and powers of the Comptroller"),
]

MULTI = [
    ("How do Articles 21 and 21A together protect life, liberty, and education?", "Article 21 protects life and personal liberty except according to procedure established by law, while Article 21A requires the State to provide free and compulsory education to children aged six to fourteen.", ["21", "21A"], ["21. Protection of life and personal liberty", "21A. Right to education"]),
    ("Compare equality before law under Article 14 with the prohibition of discrimination under Article 15.", "Article 14 guarantees equality before the law and equal protection of the laws to all persons, while Article 15 prohibits the State from discriminating against citizens on grounds only of religion, race, caste, sex, or place of birth.", ["14", "15"], ["14. Equality before law", "15. Prohibition of discrimination"]),
    ("How do Articles 19 and 20 protect expression and fair criminal process?", "Article 19 guarantees citizens freedoms such as speech and expression, peaceful assembly, association, and movement, while Article 20 protects persons from retrospective conviction and from penalties greater than those applicable when the offence was committed.", ["19", "20"], ["19. Protection of certain rights regarding freedom of speech", "20. Protection in respect of conviction"]),
    ("How do Articles 23 and 24 protect persons from exploitation?", "Article 23 prohibits traffic in human beings, begar, and similar forced labour, while Article 24 prohibits the employment of children below fourteen years in factories, mines, or other hazardous work.", ["23", "24"], ["23. Prohibition of traffic in human beings", "24. Prohibition of employment of children"]),
    ("How do Articles 29 and 30 protect minorities?", "Article 29 protects the language, script, and culture of any section of citizens and bars discrimination in admission to State-maintained or State-aided institutions; Article 30 gives religious and linguistic minorities the right to establish and administer educational institutions of their choice.", ["29", "30"], ["29. Protection of interests of minorities", "30. Right of minorities to establish"]),
    ("How do Articles 15 and 17 address caste discrimination?", "Article 15 prohibits discrimination by the State on grounds only of religion, race, caste, sex, or place of birth, while Article 17 abolishes untouchability and makes its practice an offence.", ["15", "17"], ["15. Prohibition of discrimination", "17. Abolition of Untouchability"]),
    ("How do Articles 38 and 39A promote social justice?", "Article 38 directs the State to secure a social order based on justice and to minimise inequalities, while Article 39A requires equal justice and free legal aid so that economic or other disabilities do not deny access to justice.", ["38", "39A"], ["38. State to secure a social order", "39A. Equal justice and free legal aid"]),
    ("How do Articles 40 and 41 support local self-government and welfare?", "Article 40 directs the State to organise village panchayats as units of self-government, while Article 41 requires effective provision for work, education, and public assistance within the State's economic capacity.", ["40", "41"], ["40. Organisation of village panchayats", "41. Right to work"]),
    ("How do Articles 43A and 46 protect workers and weaker sections?", "Article 43A requires steps to secure workers' participation in the management of industries, while Article 46 requires special care for the educational and economic interests of weaker sections, especially Scheduled Castes and Scheduled Tribes.", ["43A", "46"], ["43A. Participation of workers in management", "46. Promotion of educational and economic interests"]),
    ("How do Articles 47 and 48A address public health and the environment?", "Article 47 makes the raising of nutrition and living standards and the improvement of public health primary duties of the State, while Article 48A requires the State to protect and improve the environment and safeguard forests and wildlife.", ["47", "48A"], ["47. Duty of the State to raise the level of nutrition", "48A. Protection and improvement of environment"]),
    ("How do Article 49 and Article 51A protect national heritage and the environment?", "Article 49 obliges the State to protect monuments and objects of national importance, while Article 51A makes it a fundamental duty of every citizen to value heritage and to protect and improve the natural environment.", ["49", "51A"], ["49. Protection of monuments", "51A. Fundamental duties"]),
    ("How are the President and the Council of Ministers connected under Articles 52, 53, and 74?", "Article 52 establishes the office of President, Article 53 vests the executive power of the Union in the President, and Article 74 requires a Council of Ministers headed by the Prime Minister to aid and advise the President, who acts on that advice.", ["52", "53", "74"], ["52. The President of India", "53. Executive power of the Union", "74. Council of Ministers to aid and advise"]),
    ("Who elects the President under Article 54, and what executive power does Article 53 confer?", "Article 54 provides that the President is elected by an electoral college of elected members of both Houses of Parliament and elected members of State Legislative Assemblies; Article 53 vests the executive power of the Union, including supreme command of the Defence Forces, in the President.", ["54", "53"], ["54. Election of President", "53. Executive power of the Union"]),
    ("What roles do the Attorney-General under Article 76 and the Comptroller and Auditor-General under Article 148 play?", "Article 76 provides for an Attorney-General appointed by the President to advise the Government on legal matters, while Article 148 provides for a Comptroller and Auditor-General appointed by the President who audits Union and State accounts and has judicial-like removal protection.", ["76", "148"], ["76. Attorney-General for India", "148. Comptroller and Auditor-General"]),
    ("How do Article 32 and Article 141 together shape judicial remedies and precedent?", "Article 32 guarantees the right to move the Supreme Court for enforcement of fundamental rights and empowers it to issue writs, while Article 141 makes the law declared by the Supreme Court binding on all courts in India.", ["32", "141"], ["32. Remedies for enforcement of rights", "141. Law declared by Supreme Court"]),
]

AMBIGUOUS = [
    ("I was stopped from speaking at a public meeting. Which constitutional provision could protect me?", "Article 19 protects citizens' freedom of speech and expression and their right to assemble peaceably and without arms.", ["19"], ["19. Protection of certain rights regarding freedom of speech"], "The question does not name an Article; the facts point most directly to Article 19(1)(a) and (b)."),
    ("A person is held in custody without being told why. What does the Constitution require?", "Article 22 requires that an arrested person be informed of the grounds of arrest and be allowed to consult and be defended by a legal practitioner of their choice.", ["22"], ["22. Protection against arrest and detention"], "The question is fact-specific and does not name Article 22; it is answerable from that provision."),
    ("The government refused me a job because of my caste. Which provision is relevant?", "Article 16 guarantees equality of opportunity in public employment and prohibits discrimination on grounds only of religion, race, caste, sex, descent, place of birth, or residence.", ["16"], ["16. Equality of opportunity"], "Caste-based refusal in public employment points to Article 16; Article 15 may also be discussed, but 16 is the specific employment guarantee."),
    ("A place of worship refuses entry based on caste. Which constitutional provision is relevant?", "Article 17 abolishes untouchability and forbids its practice in any form, making enforcement of any disability arising from it an offence.", ["17"], ["17. Abolition of Untouchability"], "The facts do not use the word 'untouchability', but caste-based exclusion is the core concern of Article 17."),
    ("My ten-year-old child is working in a factory. Which provision protects the child?", "Article 24 prohibits the employment of children below fourteen years in any factory or mine or in other hazardous employment.", ["24"], ["24. Prohibition of employment of children"], "The age and workplace make Article 24 the relevant provision."),
    ("A minority community wants to run its own school. Which provision applies?", "Article 30 gives all minorities, whether based on religion or language, the right to establish and administer educational institutions of their choice.", ["30"], ["30. Right of minorities to establish"], "The question is vague about the type of institution; Article 30 is the specific minority educational right."),
    ("A new law takes away a fundamental right. What happens to that law?", "Under Article 13, a law that takes away or abridges fundamental rights is void to the extent of the contravention.", ["13"], ["13. Laws inconsistent with or in derogation"], "The question does not name Article 13, but the effect of an inconsistent law is governed by it."),
    ("The executive is interfering in the working of the judiciary. Which directive principle is relevant?", "Article 50 directs the State to take steps to separate the judiciary from the executive in the public services of the State.", ["50"], ["50. Separation of judiciary from executive"], "The question describes interference without naming Article 50; the separation principle is the relevant directive."),
    ("A poor person cannot afford a lawyer. Which constitutional provision is relevant?", "Article 39A requires the State to secure equal justice and provide free legal aid so that justice is not denied because of economic or other disabilities.", ["39A"], ["39A. Equal justice and free legal aid"], "The facts point to free legal aid under Article 39A; Article 21 case law is outside the corpus text."),
    ("Workers want a say in how their factory is run. Which provision applies?", "Article 43A directs the State to take steps to secure the participation of workers in the management of industries.", ["43A"], ["43A. Participation of workers in management"], "The question is broad but maps to the workers' participation directive in Article 43A."),
]

UNANSWERABLE = [
    ("What is the punishment for murder under the Indian Penal Code?", "The corpus of the Constitution of India does not provide this information.", "The Indian Penal Code is not part of the Constitution corpus."),
    ("Who is the current Chief Justice of India?", "The corpus of the Constitution of India does not provide this information.", "The corpus contains constitutional provisions, not current office-holders."),
    ("What is the capital of Australia?", "The corpus of the Constitution of India does not provide this information.", "General world geography is outside the corpus."),
    ("What are the current income tax slab rates in India?", "The corpus of the Constitution of India does not provide this information.", "Tax rates are set by statutes and annual finance acts, not the Constitution text."),
    ("What is the fee for filing a writ petition in the Supreme Court?", "The corpus of the Constitution of India does not provide this information.", "Court fees and filing procedure are not specified in the constitutional text."),
    ("What is the procedure for incorporating a company in India?", "The corpus of the Constitution of India does not provide this information.", "Company law procedure is outside the corpus."),
    ("Who won the 2024 Indian general election?", "The corpus of the Constitution of India does not provide this information.", "Election results are not part of the constitutional text."),
    ("What is the speed limit on Indian national highways?", "The corpus of the Constitution of India does not provide this information.", "Traffic rules are statutory/administrative, not constitutional."),
    ("What is the procedure for divorce under the Hindu Marriage Act?", "The corpus of the Constitution of India does not provide this information.", "Personal law statutes are outside the corpus."),
    ("What is the punishment for theft under the Indian Penal Code?", "The corpus of the Constitution of India does not provide this information.", "The Indian Penal Code is not part of the corpus."),
    ("What is the minimum wage in Delhi?", "The corpus of the Constitution of India does not provide this information.", "Minimum wage rates are not fixed in the Constitution text."),
    ("How do I apply for an Indian passport?", "The corpus of the Constitution of India does not provide this information.", "Passport procedure is administrative and outside the corpus."),
    ("Which country won the 2023 Cricket World Cup?", "The corpus of the Constitution of India does not provide this information.", "Sports results are outside the corpus."),
    ("What is the recipe for biryani?", "The corpus of the Constitution of India does not provide this information.", "Culinary information is outside the corpus."),
    ("Which engineering college is the best in India?", "The corpus of the Constitution of India does not provide this information.", "Rankings and opinions are outside the corpus."),
    ("What is the medical treatment for diabetes?", "The corpus of the Constitution of India does not provide this information.", "Medical advice is outside the corpus."),
    ("Will the stock market rise next week?", "The corpus of the Constitution of India does not provide this information.", "Predictions are outside the corpus."),
    ("What punishment is prescribed for failing to perform fundamental duties?", "The corpus of the Constitution of India does not provide this information.", "Article 51A lists duties but the corpus text does not prescribe a penalty for non-performance."),
    ("What is the population of India?", "The corpus of the Constitution of India does not provide this information.", "Current demographic statistics are outside the corpus."),
    ("Who is the Governor of Maharashtra?", "The corpus of the Constitution of India does not provide this information.", "Current office-holders are not named in the corpus text."),
]

DIRECT_TEMPLATES = [
    "What does Article {n} of the Constitution of India provide?",
    "State the constitutional rule contained in Article {n}.",
    "According to the Constitution of India, what is provided by Article {n}?",
]

PARAPHRASE_TEMPLATES = [
    "Can you explain the constitutional position on {topic}?",
    "What does the Constitution say about {topic}?",
    "Summarise the constitutional principle concerning {topic}.",
]

KEYWORD_TEMPLATES = [
    "Which Article deals with {keyword}, and what does it say?",
    "Under the Constitution of India, what is the rule regarding {keyword}?",
    "What does Article {n} say about {keyword}?",
]


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {path}. Run extract_pdf.py, clean_text.py, "
            "and chunk_text.py first."
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_evidence_list(chunks: list[dict], headings: list[str]) -> list[str]:
    resolved = [resolve_evidence(chunks, heading) for heading in headings]
    return list(dict.fromkeys(resolved))


def resolve_evidence(chunks: list[dict], heading: str) -> str:
    matches = []
    for chunk in chunks:
        if chunk["page_start"] < 30:
            continue
        index = chunk["text"].find(heading)
        if index != -1 and "—" in chunk["text"][index:index + 250]:
            matches.append(chunk["chunk_id"])
    if not matches:
        raise ValueError(f"Evidence heading not found in body corpus: {heading!r}")
    if len(matches) > 1:
        raise ValueError(f"Evidence heading is ambiguous ({matches}): {heading!r}")
    return matches[0]


def build_rows(chunks: list[dict]) -> list[dict]:
    rows = []
    for i, (art, answer, topic, keyword, heading) in enumerate(FACTS):
        chunk_id = resolve_evidence(chunks, heading)
        rows.append({
            "question": DIRECT_TEMPLATES[i % len(DIRECT_TEMPLATES)].format(n=art),
            "reference_answer": answer,
            "relevant_articles": [art],
            "evidence_chunk_ids": [chunk_id],
            "answerable_from_corpus": True,
            "category": "direct_article_question",
            "notes": "",
        })
        rows.append({
            "question": PARAPHRASE_TEMPLATES[i % len(PARAPHRASE_TEMPLATES)].format(topic=topic),
            "reference_answer": answer,
            "relevant_articles": [art],
            "evidence_chunk_ids": [chunk_id],
            "answerable_from_corpus": True,
            "category": "paraphrased_question",
            "notes": "",
        })
        rows.append({
            "question": KEYWORD_TEMPLATES[i % len(KEYWORD_TEMPLATES)].format(n=art, keyword=keyword),
            "reference_answer": answer,
            "relevant_articles": [art],
            "evidence_chunk_ids": [chunk_id],
            "answerable_from_corpus": True,
            "category": "keyword_heavy_question",
            "notes": "",
        })

    for question, answer, articles, headings in MULTI:
        rows.append({
            "question": question,
            "reference_answer": answer,
            "relevant_articles": articles,
            "evidence_chunk_ids": resolve_evidence_list(chunks, headings),
            "answerable_from_corpus": True,
            "category": "multi_article_question",
            "notes": "Requires evidence from more than one Article.",
        })

    for question, answer, articles, headings, note in AMBIGUOUS:
        rows.append({
            "question": question,
            "reference_answer": answer,
            "relevant_articles": articles,
            "evidence_chunk_ids": resolve_evidence_list(chunks, headings),
            "answerable_from_corpus": True,
            "category": "ambiguous_question",
            "notes": note,
        })

    for question, answer, note in UNANSWERABLE:
        rows.append({
            "question": question,
            "reference_answer": answer,
            "relevant_articles": [],
            "evidence_chunk_ids": [],
            "answerable_from_corpus": False,
            "category": "unanswerable_out_of_corpus",
            "notes": note,
        })

    return rows


def assign_ids_and_splits(rows: list[dict]) -> None:
    for i, row in enumerate(rows, start=1):
        row["question_id"] = f"q{i:03d}"

    dev_targets = {
        category: math.ceil(sum(r["category"] == category for r in rows) * DEV_FRACTION)
        for category in CATEGORY_ORDER
    }
    dev_used = {category: 0 for category in CATEGORY_ORDER}
    for row in rows:
        category = row["category"]
        if dev_used[category] < dev_targets[category]:
            row["split"] = "dev"
            dev_used[category] += 1
        else:
            row["split"] = "test"
        row["review_status"] = "draft_pending_human_review"

    for row in rows:
        ordered = {
            "question_id": row["question_id"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "relevant_articles": row["relevant_articles"],
            "evidence_chunk_ids": row["evidence_chunk_ids"],
            "answerable_from_corpus": row["answerable_from_corpus"],
            "category": row["category"],
            "split": row["split"],
            "notes": row["notes"],
            "review_status": row["review_status"],
        }
        row.clear()
        row.update(ordered)


def main() -> None:
    chunks = load_chunks()
    rows = build_rows(chunks)
    assign_ids_and_splits(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    split_counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
        split_counts[(row["category"], row["split"])] = (
            split_counts.get((row["category"], row["split"]), 0) + 1
        )

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    for category in CATEGORY_ORDER:
        print(
            f"  {category}: {counts.get(category, 0)} "
            f"(dev={split_counts.get((category, 'dev'), 0)}, "
            f"test={split_counts.get((category, 'test'), 0)})"
        )
    print(
        "  split totals: dev=", sum(r["split"] == "dev" for r in rows),
        " test=", sum(r["split"] == "test" for r in rows),
    )


if __name__ == "__main__":
    main()
