from research_extension.lexical import LexicalIndex


def test_postings_df_positions_and_field_lengths(synthetic_index):
    assert synthetic_index.document_count == 5
    assert synthetic_index.document_frequency("quantum") == 2
    quantum = synthetic_index.postings["quantum"]
    assert quantum[0].doc_index == 0
    assert quantum[0].title_positions == (0,)
    assert quantum[0].body_positions == ()
    assert synthetic_index.title_lengths[0] == 2
    assert synthetic_index.body_lengths[0] == 3
    assert synthetic_index.title_lengths[4] == 0
    assert synthetic_index.body_lengths[4] == 0


def test_no_cross_document_position_contamination(synthetic_index):
    red = synthetic_index.postings["red"]
    assert len(red) == 2
    assert red[0].doc_index == 2
    assert red[0].title_positions == (0,)
    assert red[0].body_positions == (0,)
    assert red[1].doc_index == 3
    assert red[1].body_positions == (0,)


def test_serialization_round_trip(synthetic_index, tmp_path):
    path = tmp_path / "lexical.pkl"
    synthetic_index.save(path)
    restored = LexicalIndex.load(path)
    assert restored.doc_ids == synthetic_index.doc_ids
    assert restored.postings == synthetic_index.postings
    assert restored.body_lengths == synthetic_index.body_lengths
