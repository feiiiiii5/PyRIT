# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import io
from io import StringIO

from pyrit.common.csv_helper import read_csv, strip_leading_comment_block, write_csv


def test_read_csv_returns_list_of_dicts():
    data = "name,age\nAlice,30\nBob,25\n"
    result = read_csv(StringIO(data))
    assert result == [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]


def test_read_csv_empty_body():
    data = "name,age\n"
    result = read_csv(StringIO(data))
    assert result == []


def test_read_csv_single_column():
    data = "value\nfoo\nbar\n"
    result = read_csv(StringIO(data))
    assert result == [{"value": "foo"}, {"value": "bar"}]


def test_write_csv_produces_expected_output():
    output = StringIO()
    examples = [{"col1": "a", "col2": "b"}, {"col1": "c", "col2": "d"}]
    write_csv(output, examples)
    output.seek(0)
    lines = output.read().strip().splitlines()
    assert lines[0] == "col1,col2"
    assert lines[1] == "a,b"
    assert lines[2] == "c,d"


def test_write_then_read_roundtrip():
    examples = [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}]
    buf = StringIO()
    write_csv(buf, examples)
    buf.seek(0)
    result = read_csv(buf)
    assert result == examples


def test_write_csv_empty_examples_writes_nothing():
    file = io.StringIO()
    write_csv(file, [])
    assert file.getvalue() == ""


def test_write_csv_writes_header_and_rows():
    file = io.StringIO()
    write_csv(file, [{"name": "alice", "role": "admin"}])
    lines = file.getvalue().strip().splitlines()
    assert lines[0] == "name,role"
    assert lines[1] == "alice,admin"


def test_strip_leading_comment_block_preserves_crlf_and_cr_inside_quoted_cells():
    text = '# dataset_version=1.0\r\nobjective,assistant_response\r\nplain,"line one\r\n### heading\rline three"\r\n'
    body = strip_leading_comment_block(text)
    assert body == 'objective,assistant_response\r\nplain,"line one\r\n### heading\rline three"\r\n'


def test_strip_leading_comment_block_handles_cr_only_record_separators():
    text = "# dataset_version=1.0\robjective,assistant_response\rplain,refusal response\r"
    body = strip_leading_comment_block(text)
    assert body == "objective,assistant_response\rplain,refusal response\r"
    assert list(csv.DictReader(StringIO(body, newline=""))) == [
        {"objective": "plain", "assistant_response": "refusal response"}
    ]
