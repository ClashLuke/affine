import pytest

from affine.vllm import LocalSlots, _parse_field


# --- _parse_field ---

def test_parse_field_basic():
    assert _parse_field("App: abc123\nUrl: http://host:8000", "app") == "abc123"


def test_parse_field_case_insensitive():
    assert _parse_field("APP: abc123", "app") == "abc123"
    assert _parse_field("app: abc123", "App") == "abc123"


def test_parse_field_colon_in_value():
    assert _parse_field("Url: http://host:8000/v1", "url") == "http://host:8000/v1"


def test_parse_field_missing_raises():
    with pytest.raises(RuntimeError, match="field 'app' not found"):
        _parse_field("Url: http://host:8000", "app")


# --- LocalSlots ---

@pytest.mark.asyncio
async def test_local_slots_lifecycle():
    slots = LocalSlots("http://a/v1", "http://b/v1")
    s1 = await slots.provision("m1", "r1")
    assert s1.base_url == "http://a/v1"
    s2 = await slots.provision("m2", "r2")
    assert s2.base_url == "http://b/v1"
    await slots.teardown(s1)
    s3 = await slots.provision("m3", "r3")
    assert s3.base_url == "http://a/v1"


@pytest.mark.asyncio
async def test_local_slots_exhaustion():
    slots = LocalSlots("http://a/v1", "http://b/v1")
    await slots.provision("m1", "r1")
    await slots.provision("m2", "r2")
    with pytest.raises(RuntimeError, match="no free local slots"):
        await slots.provision("m3", "r3")
