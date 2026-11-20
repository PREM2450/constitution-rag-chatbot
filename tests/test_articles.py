import unittest

from chunk_text import build_chunks, build_supplement_chunks
from create_embeddings import embedding_text
from extract_articles import parse_articles


class ArticleExtractionTests(unittest.TestCase):
    def test_skips_contents_and_page_headers_but_keeps_wrapped_article_headings(self):
        pages = [
            {
                "page_number": 1,
                "text": "ARTICLES\nContents\n1. Name and territory of the Union.\n",
            },
            {
                "page_number": 2,
                "text": "THE CONSTITUTION OF INDIA\nPREAMBLE\n"
                "WE, THE PEOPLE OF INDIA, having solemnly resolved...",
            },
            {
                "page_number": 3,
                "text": "PART I\n1. Name and territory of the Union.—India is a Union.\n",
            },
            {
                "page_number": 4,
                "text": "33\nTHE CONSTITUTION OF INDIA\n(Part I.—Territory)\n"
                "2[33. Power of Parliament to modify the rights conferred by this Part\n"
                "in their application to Forces, etc.—Parliament may determine.\n",
            },
            {
                "page_number": 5,
                "text": "395. Repeals.—Earlier laws are repealed.\n"
                "1[FIRST SCHEDULE\n1. States\n",
            },
        ]

        articles = parse_articles(pages)
        by_number = {article["article_number"]: article for article in articles}

        self.assertEqual(list(by_number), ["1", "33", "395"])
        self.assertEqual(by_number["1"]["title"], "Name and territory of the Union.")
        self.assertEqual(by_number["33"]["part"], "Part I")
        self.assertIn("Parliament may determine", by_number["33"]["text"])
        self.assertNotIn("States", by_number["395"]["text"])

    def test_parses_headings_wrapped_over_many_lines(self):
        article = parse_articles([
            {"page_number": 1, "text": "WE, THE PEOPLE OF INDIA"},
            {
                "page_number": 2,
                "text": "136. Special\nleave to appeal by\nthe Supreme Court.—"
                "The Supreme Court may grant leave.",
            },
        ])[0]

        self.assertEqual(article["article_number"], "136")
        self.assertEqual(article["title"], "Special leave to appeal by the Supreme Court.")
        self.assertEqual(article["text"], "The Supreme Court may grant leave.")


class ArticleChunkingTests(unittest.TestCase):
    def test_article_metadata_survives_chunking(self):
        article = {
            "article_number": "21",
            "article": "Article 21",
            "title": "Protection of life and personal liberty.",
            "text": "A " * 40,
            "part": "Part III",
            "page_numbers": [42, 43],
            "source_title": "The Constitution of India",
            "source_url": "https://example.invalid/constitution.pdf",
            "source_version": "2026-09-10",
        }

        chunks = build_chunks([article], chunk_size=20, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "article_21_chunk_0")
        self.assertTrue(all(chunk["article"] == "Article 21" for chunk in chunks))
        self.assertTrue(all(chunk["page_start"] == 42 for chunk in chunks))
        self.assertTrue(all(chunk["page_end"] == 43 for chunk in chunks))
        self.assertTrue(all(chunk["source_version"] == "2026-09-10" for chunk in chunks))

    def test_embedding_text_includes_article_heading(self):
        text = embedding_text({
            "article": "Article 21",
            "title": "Protection of life and personal liberty.",
            "text": "No person shall be deprived of life.",
        })

        self.assertIn("Article 21", text)
        self.assertIn("Protection of life", text)
        self.assertIn("No person shall", text)

    def test_schedules_and_appendices_are_page_scoped_chunks(self):
        chunks = build_supplement_chunks(
            [
                {"page_number": 10, "text": "Article 395 content"},
                {"page_number": 11, "text": "1[FIRST SCHEDULE\nStates and territories"},
                {"page_number": 12, "text": "APPENDIX I\nAdditional material"},
            ],
            after_page=10,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section"], "First Schedule")
        self.assertEqual(chunks[0]["page_start"], 11)
        self.assertEqual(chunks[1]["section"], "Appendices")
        self.assertIsNone(chunks[1]["article"])


if __name__ == "__main__":
    unittest.main()
