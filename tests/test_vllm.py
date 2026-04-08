import pytest
from affine.vllm import LocalSlots, Slot, _parse_field


@pytest.mark.asyncio
async def test_local_slots_provision_teardown_cycle():
    slots = LocalSlots("http://a:8000/v1", "http://b:8001/v1")
    s1 = await slots.provision("m1", "r1")
    assert s1.base_url == "http://a:8000/v1"
    s2 = await slots.provision("m2", "r2")
    assert s2.base_url == "http://b:8001/v1"

    # No free slots
    with pytest.raises(RuntimeError, match="no free"):
        await slots.provision("m3", "r3")

    # Teardown returns URL to pool
    await slots.teardown(s1)
    s3 = await slots.provision("m3", "r3")
    assert s3.base_url == "http://a:8000/v1"


@pytest.mark.asyncio
async def test_local_slots_dethrone_cycle():
    """Simulate a full dethrone: provision champion + challenger, teardown champion, provision new challenger."""
    slots = LocalSlots("http://a:8000/v1", "http://b:8001/v1")
    champion = await slots.provision("champ", "r1")
    challenger = await slots.provision("chall", "r1")

    # Dethrone: teardown old champion
    await slots.teardown(champion)

    # New challenger gets the freed URL
    new_chall = await slots.provision("chall2", "r1")
    assert new_chall.base_url == "http://a:8000/v1"
    assert challenger.base_url == "http://b:8001/v1"


@pytest.mark.asyncio
async def test_local_slots_double_teardown():
    slots = LocalSlots("http://a:8000/v1", "http://b:8001/v1")
    s = await slots.provision("m", "r")
    await slots.teardown(s)
    await slots.teardown(s)  # should not add URL twice
    assert len(slots._free) == 2


def test_parse_field_app_id():
    output = "Created successfully\nApp ID: app-abc123\nStatus: running\n"
    assert _parse_field(output, "app") == "app-abc123"


def test_parse_field_url():
    output = "App ID: app-abc\nURL: https://fnc-xxx.serverless.targon.com\n"
    assert _parse_field(output, "url") == "https://fnc-xxx.serverless.targon.com"


def test_parse_field_missing():
    with pytest.raises(RuntimeError, match="not found"):
        _parse_field("no relevant output here", "app")


def test_parse_field_url_with_port():
    output = "URL: https://example.com:8080/path"
    # The split(":", 1) will get "URL" and " https://example.com:8080/path"
    assert _parse_field(output, "url") == "https://example.com:8080/path"
