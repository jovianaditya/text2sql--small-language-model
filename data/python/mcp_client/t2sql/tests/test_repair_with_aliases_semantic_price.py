import unittest
from unittest.mock import patch

from t2sql import helpers_main


class _FakeAnalyzer:
    def __init__(self, table_columns):
        self._table_columns = table_columns

    def get_table_columns(self, table):
        return list(self._table_columns.get(table, ()))


class RepairWithAliasesSemanticPriceTest(unittest.TestCase):
    def test_maps_price_aliases_from_metadata_without_classicmodels_names(self):
        analyzer = _FakeAnalyzer(
            {
                "inventory_items": ["itemId", "catalogAmount"],
                "sales_lines": ["saleId", "lineCharge"],
            }
        )
        synonyms = {
            ("inventory_items", "catalogAmount"): ["list price", "retail price"],
            ("sales_lines", "lineCharge"): ["unit price", "selling price"],
        }
        ctx = """### ALLOWED IDENTIFIERS
- inventory_items: itemId, catalogAmount
- sales_lines: saleId, lineCharge
"""

        with patch.object(helpers_main, "get_schema_analyzer", return_value=analyzer), patch.object(
            helpers_main, "get_all_tables", return_value=["inventory_items", "sales_lines"]
        ), patch.object(helpers_main, "get_name_column", return_value=None), patch.object(
            helpers_main, "get_id_column", return_value=None
        ), patch.object(
            helpers_main,
            "get_column_synonyms",
            side_effect=lambda table, col: list(synonyms.get((table, col), ())),
        ):
            repaired = helpers_main._repair_with_aliases(
                "SELECT listprice, price FROM sales_lines;",
                ctx,
            )

        self.assertEqual(
            repaired,
            "SELECT catalogAmount, lineCharge FROM sales_lines;",
        )

    def test_leaves_generic_price_unmapped_when_metadata_is_ambiguous(self):
        analyzer = _FakeAnalyzer(
            {
                "sales_lines": ["saleId", "lineCharge"],
                "quote_lines": ["quoteId", "quotedAmount"],
            }
        )
        synonyms = {
            ("sales_lines", "lineCharge"): ["unit price"],
            ("quote_lines", "quotedAmount"): ["unit price"],
        }
        ctx = """### ALLOWED IDENTIFIERS
- sales_lines: saleId, lineCharge
- quote_lines: quoteId, quotedAmount
"""

        with patch.object(helpers_main, "get_schema_analyzer", return_value=analyzer), patch.object(
            helpers_main, "get_all_tables", return_value=["sales_lines", "quote_lines"]
        ), patch.object(helpers_main, "get_name_column", return_value=None), patch.object(
            helpers_main, "get_id_column", return_value=None
        ), patch.object(
            helpers_main,
            "get_column_synonyms",
            side_effect=lambda table, col: list(synonyms.get((table, col), ())),
        ):
            repaired = helpers_main._repair_with_aliases(
                "SELECT price FROM sales_lines;",
                ctx,
            )

        self.assertEqual(repaired, "SELECT price FROM sales_lines;")


if __name__ == "__main__":
    unittest.main()
