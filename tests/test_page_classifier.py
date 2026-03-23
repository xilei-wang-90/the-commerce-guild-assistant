"""Tests for guild_assistant.utils.page_classifier."""

from pathlib import Path

import pytest

from guild_assistant.utils.page_classifier import (
    PageType,
    classify_file,
    classify_page,
    extract_heading_titles,
)


# ---------------------------------------------------------------------------
# classify_page (re-exported from utils)
# ---------------------------------------------------------------------------


class TestClassifyPage:
    def test_item_page(self):
        assert PageType.ITEM in classify_page("yakmel.md", ["Obtaining", "Usage"])

    def test_location_page(self):
        assert PageType.LOCATION in classify_page("eufaula.md", ["Region", "NPCs"])

    def test_character_page(self):
        assert PageType.CHARACTER in classify_page("mi_an.md", ["Biographical Information"])

    def test_monster_page(self):
        assert PageType.MONSTER in classify_page("pensky.md", ["Battle Statistics"])

    def test_mission_by_filename(self):
        assert PageType.MISSION in classify_page("mission_x.md", ["Walkthrough"])

    def test_dialogue_by_filename(self):
        assert PageType.DIALOGUE in classify_page("mi_an_dialogue.md", ["Greetings"])

    def test_buyback_by_filename(self):
        assert PageType.BUYBACK in classify_page("yakmel_buyback.md", [])

    def test_event_by_filename(self):
        assert PageType.EVENT in classify_page("event_sandstorm.md", ["Rewards"])

    def test_store_page(self):
        assert PageType.STORE in classify_page("shop.md", ["Stock", "Hours"])

    def test_region_page(self):
        assert PageType.REGION in classify_page("sandrock.md", ["Population"])

    def test_festival_page(self):
        assert PageType.FESTIVAL in classify_page("day.md", ["Time", "Activities"])

    def test_generic_page(self):
        assert classify_page("random.md", ["Trivia"]) == [PageType.GENERIC]

    def test_filename_priority(self):
        assert classify_page("mission_x.md", ["Obtaining"]) == [PageType.MISSION]


# ---------------------------------------------------------------------------
# extract_heading_titles
# ---------------------------------------------------------------------------


class TestExtractHeadingTitles:
    def test_atx_headings(self):
        titles = extract_heading_titles("# Title\n## Section\n### Sub")
        assert "Title" in titles
        assert "Section" in titles
        assert "Sub" in titles

    def test_setext_headings(self):
        titles = extract_heading_titles("Title\n=====\n\nSection\n------")
        assert "Title" in titles
        assert "Section" in titles

    def test_empty_content(self):
        assert extract_heading_titles("") == []

    def test_no_headings(self):
        assert extract_heading_titles("Just plain text.") == []


# ---------------------------------------------------------------------------
# classify_file
# ---------------------------------------------------------------------------


class TestClassifyFile:
    def test_item_file(self, tmp_path):
        f = tmp_path / "sword.md"
        f.write_text("# Sword\n## Obtaining\nCraft it.", encoding="utf-8")
        assert PageType.ITEM in classify_file(f)

    def test_character_file(self, tmp_path):
        f = tmp_path / "mi_an.md"
        f.write_text("# Mi-an\n## Biographical Information\nAge: 18.", encoding="utf-8")
        assert PageType.CHARACTER in classify_file(f)

    def test_mission_file_by_name(self, tmp_path):
        f = tmp_path / "mission_test.md"
        f.write_text("# Mission\nDo stuff.", encoding="utf-8")
        assert PageType.MISSION in classify_file(f)
