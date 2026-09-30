import json

from prompt_cache_meter.proxy.replies import StreamCollector, summarise_completion


def test_stream_line_split_across_chunks_is_still_read() -> None:
    line = f"data: {json.dumps({'choices': [{'delta': {'content': 'hello'}}]})}\n\n".encode()
    collector = StreamCollector()

    collector.feed(line[:10])
    collector.feed(line[10:])

    assert collector.summary().text == "hello"


def test_tool_call_arguments_count_as_output() -> None:
    call = {"function": {"name": "lookup", "arguments": '{"id": 7}'}}
    body = json.dumps({"choices": [{"message": {"content": None, "tool_calls": [call]}}]}).encode()

    assert summarise_completion(body).text == 'lookup{"id": 7}'


def test_non_json_reply_gives_empty_summary() -> None:
    summary = summarise_completion(b"<html>oops</html>")

    assert summary.text == ""
    assert summary.usage is None
