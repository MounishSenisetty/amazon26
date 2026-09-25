from src.tsv import MATCH_HEADER, read_id_lists, read_tsv, write_id_lists


def test_round_trip_keeps_empty_lists_and_order(tmp_path):
    path = tmp_path / "matching_results.tsv"
    lists = {"S1-00001": ["S2-00047", "S3-00812", "S2-00047"], "S1-00003": []}
    write_id_lists(path, MATCH_HEADER, ["S1-00003", "S1-00001", "S1-00002"], lists)
    assert path.read_text() == (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00003\t\n"
        "S1-00001\tS2-00047,S3-00812\n"
        "S1-00002\t\n"
    )
    assert read_id_lists(path) == {"S1-00003": [], "S1-00001": ["S2-00047", "S3-00812"], "S1-00002": []}


def test_read_keeps_commas_quotes_and_na_literal(tmp_path):
    path = tmp_path / "s.tsv"
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        'S1-00001\tNA\t12 "Old" Mill Rd, Pune, 411001\tIndia\n'
    )
    row = read_tsv(path).iloc[0]
    assert row["business_name"] == "NA"
    assert row["business_address"] == '12 "Old" Mill Rd, Pune, 411001'
