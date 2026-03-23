"""Tests for guild_assistant.rag_setup.section_breaker."""

from pathlib import Path

import pytest

from guild_assistant.rag_setup.section_breaker import (
    SectionBreaker,
    _extract_preamble_and_blocks,
    _find_headings,
    _has_non_heading_text,
    _overview_block_indices,
    _title_to_slug,
)
from guild_assistant.utils.page_classifier import PageType, classify_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate_input(input_dir: Path, files: dict[str, str]) -> None:
    """Write *files* mapping ``{name: content}`` into *input_dir*."""
    input_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (input_dir / name).write_text(content, encoding="utf-8")


def _blocks_to_titles(blocks):
    """Extract titles from L2 blocks for easy assertion."""
    return [b.title for b in blocks]


# ---------------------------------------------------------------------------
# _title_to_slug
# ---------------------------------------------------------------------------


class TestTitleToSlug:
    def test_simple_title(self):
        assert _title_to_slug("Obtaining") == "obtaining"

    def test_multi_word(self):
        assert _title_to_slug("Quest Rewards") == "quest_rewards"

    def test_special_characters(self):
        assert _title_to_slug("Grace (NPC)") == "grace_npc"

    def test_hyphen_replaced(self):
        assert _title_to_slug("Mid-Autumn Festival") == "mid_autumn_festival"

    def test_leading_trailing_stripped(self):
        assert _title_to_slug("  Hello World  ") == "hello_world"

    def test_numbers_preserved(self):
        assert _title_to_slug("Level 3 Upgrade") == "level_3_upgrade"


# ---------------------------------------------------------------------------
# _find_headings
# ---------------------------------------------------------------------------


class TestFindHeadings:
    def test_atx_headings(self):
        headings = _find_headings("# Title\n## Section\n### Sub")
        assert len(headings) == 3
        assert headings[0].level == 1
        assert headings[0].title == "Title"
        assert headings[1].level == 2
        assert headings[2].level == 3

    def test_setext_h1_with_equals(self):
        headings = _find_headings("Title\n=====\nBody text.")
        assert len(headings) == 1
        assert headings[0].level == 1
        assert headings[0].title == "Title"

    def test_setext_h2_with_dashes(self):
        headings = _find_headings("Section\n-------\nBody text.")
        assert len(headings) == 1
        assert headings[0].level == 2
        assert headings[0].title == "Section"

    def test_setext_requires_text_above(self):
        """A --- line after a blank line is a thematic break, not a heading."""
        headings = _find_headings("Some text.\n\n---\nMore text.")
        assert len(headings) == 0

    def test_setext_minimum_two_dashes(self):
        headings = _find_headings("Section\n--\nBody.")
        assert len(headings) == 1
        assert headings[0].level == 2

    def test_mixed_atx_and_setext(self):
        content = "# ATX Title\nIntro.\n\nSetext Section\n---\nBody."
        headings = _find_headings(content)
        assert len(headings) == 2
        assert headings[0].level == 1
        assert headings[0].title == "ATX Title"
        assert headings[1].level == 2
        assert headings[1].title == "Setext Section"

    def test_setext_end_position_after_underline(self):
        content = "Title\n===\nBody here."
        headings = _find_headings(content)
        assert len(headings) == 1
        assert content[headings[0].end :].strip() == "Body here."

    def test_thematic_break_not_detected_as_heading(self):
        headings = _find_headings("---\nSome text.")
        assert len(headings) == 0

    def test_setext_trailing_spaces_on_underline(self):
        headings = _find_headings("Title\n===   \nBody.")
        assert len(headings) == 1
        assert headings[0].level == 1


# ---------------------------------------------------------------------------
# _has_non_heading_text
# ---------------------------------------------------------------------------


class TestHasNonHeadingText:
    def test_plain_text(self):
        assert _has_non_heading_text("Hello world") is True

    def test_only_atx_headings(self):
        assert _has_non_heading_text("## Section\n### Sub") is False

    def test_heading_plus_text(self):
        assert _has_non_heading_text("## Section\nSome text.") is True

    def test_only_setext_heading(self):
        assert _has_non_heading_text("Section\n------") is False

    def test_empty_string(self):
        assert _has_non_heading_text("") is False

    def test_only_whitespace(self):
        assert _has_non_heading_text("  \n\n  ") is False

    def test_setext_plus_text(self):
        assert _has_non_heading_text("Section\n------\nBody.") is True


# ---------------------------------------------------------------------------
# _extract_preamble_and_blocks
# ---------------------------------------------------------------------------


class TestExtractPreambleAndBlocks:
    def test_no_headings(self):
        preamble, blocks = _extract_preamble_and_blocks("Just text.", [])
        assert preamble == "Just text."
        assert blocks == []

    def test_only_l1(self):
        content = "# Title\nIntro text."
        headings = _find_headings(content)
        preamble, blocks = _extract_preamble_and_blocks(content, headings)
        assert "Intro text." in preamble
        assert blocks == []

    def test_l1_and_l2(self):
        content = "# Title\nIntro.\n## Section A\nText A."
        headings = _find_headings(content)
        preamble, blocks = _extract_preamble_and_blocks(content, headings)
        assert "Intro." in preamble
        assert len(blocks) == 1
        assert blocks[0].title == "Section A"
        assert "Text A." in blocks[0].content

    def test_multiple_l2(self):
        content = "## A\nText A.\n## B\nText B.\n## C\nText C."
        headings = _find_headings(content)
        preamble, blocks = _extract_preamble_and_blocks(content, headings)
        assert preamble == ""
        assert _blocks_to_titles(blocks) == ["A", "B", "C"]

    def test_l3_included_in_l2_block(self):
        content = "## A\nText A.\n### A1\nText A1.\n## B\nText B."
        headings = _find_headings(content)
        preamble, blocks = _extract_preamble_and_blocks(content, headings)
        assert len(blocks) == 2
        assert "### A1" in blocks[0].content
        assert "Text A1." in blocks[0].content

    def test_orphan_l3_goes_to_preamble(self):
        content = "### Orphan\nOrphan text.\n## A\nText A."
        headings = _find_headings(content)
        preamble, blocks = _extract_preamble_and_blocks(content, headings)
        assert "Orphan" in preamble
        assert len(blocks) == 1

    def test_content_before_heading_in_preamble(self):
        content = "Preamble text.\n## Section\nSection text."
        headings = _find_headings(content)
        preamble, blocks = _extract_preamble_and_blocks(content, headings)
        assert "Preamble text." in preamble
        assert len(blocks) == 1

    def test_l1_between_l2_closes_block(self):
        content = "## A\nText A.\n# Title\nTitle text.\n## B\nText B."
        headings = _find_headings(content)
        preamble, blocks = _extract_preamble_and_blocks(content, headings)
        assert len(blocks) == 2
        assert blocks[0].title == "A"
        assert blocks[1].title == "B"
        assert "Title text." in preamble

    def test_l2_block_includes_header_line(self):
        content = "## Obtaining\nYou can get one."
        headings = _find_headings(content)
        _, blocks = _extract_preamble_and_blocks(content, headings)
        assert blocks[0].content.startswith("## Obtaining")


# ---------------------------------------------------------------------------
# classify_page
# ---------------------------------------------------------------------------


class TestClassifyPage:
    def test_item_page(self):
        types = classify_page("yakmel.md", ["Obtaining", "Usage"])
        assert PageType.ITEM in types

    def test_location_page(self):
        types = classify_page("eufaula_salvage.md", ["Region", "NPCs"])
        assert PageType.LOCATION in types

    def test_character_page(self):
        types = classify_page("mi_an.md", ["Biographical Information", "Gifts"])
        assert PageType.CHARACTER in types

    def test_character_page_physical_desc(self):
        types = classify_page("mi_an.md", ["Physical Description"])
        assert PageType.CHARACTER in types

    def test_monster_page(self):
        types = classify_page("pensky.md", ["Battle Statistics", "Drops"])
        assert PageType.MONSTER in types

    def test_mission_page_by_filename(self):
        types = classify_page("mission_the_bright_sun.md", ["Walkthrough"])
        assert PageType.MISSION in types

    def test_mission_with_battle_statistics_not_monster(self):
        types = classify_page(
            "mission_sandrock_strikes_back.md", ["Battle Statistics", "Conduct"]
        )
        assert PageType.MISSION in types
        assert PageType.MONSTER not in types

    def test_dialogue_page_by_filename(self):
        types = classify_page("mi_an_dialogue.md", ["Greetings"])
        assert PageType.DIALOGUE in types

    def test_buyback_page_by_filename(self):
        types = classify_page("yakmel_buyback.md", [])
        assert PageType.BUYBACK in types

    def test_event_page_by_filename(self):
        types = classify_page("event_sandstorm.md", ["Rewards"])
        assert PageType.EVENT in types

    def test_store_page(self):
        types = classify_page("by_the_stairs.md", ["Stock", "Hours"])
        assert PageType.STORE in types

    def test_region_page(self):
        types = classify_page("sandrock.md", ["Population", "History"])
        assert PageType.REGION in types

    def test_festival_page(self):
        types = classify_page("day_of_memories.md", ["Time", "Activities"])
        assert PageType.FESTIVAL in types

    def test_generic_page(self):
        types = classify_page("random.md", ["Trivia", "Gallery"])
        assert types == [PageType.GENERIC]

    def test_filename_takes_priority_over_content(self):
        types = classify_page("mission_escort.md", ["Obtaining", "Walkthrough"])
        assert types == [PageType.MISSION]

    def test_case_insensitive_headings(self):
        types = classify_page("x.md", ["obtaining"])
        assert PageType.ITEM in types


# ---------------------------------------------------------------------------
# _overview_block_indices
# ---------------------------------------------------------------------------


class TestOverviewBlockIndices:
    def _make_blocks(self, titles):
        from guild_assistant.rag_setup.section_breaker import _L2Block
        return [_L2Block(title=t, content=f"## {t}\nContent.") for t in titles]

    def test_general_overview_sections(self):
        blocks = self._make_blocks(["Overview", "Stats", "Gallery"])
        result = _overview_block_indices(blocks, [PageType.GENERIC])
        assert result == {0}

    def test_information_in_overview(self):
        blocks = self._make_blocks(["Information", "Gallery"])
        result = _overview_block_indices(blocks, [PageType.GENERIC])
        assert result == {0}

    def test_general_information_in_overview(self):
        blocks = self._make_blocks(["General Information", "Gallery"])
        result = _overview_block_indices(blocks, [PageType.GENERIC])
        assert result == {0}

    def test_everything_before_overview_section(self):
        blocks = self._make_blocks(["Stats", "Info", "Overview", "Gallery"])
        result = _overview_block_indices(blocks, [PageType.GENERIC])
        # Everything up to and including "Overview" (indices 0, 1, 2)
        assert result == {0, 1, 2}

    def test_item_before_obtaining(self):
        blocks = self._make_blocks(["Stats", "Info", "Obtaining", "Gallery"])
        result = _overview_block_indices(blocks, [PageType.ITEM])
        # Stats and Info are before Obtaining
        assert 0 in result
        assert 1 in result
        assert 2 not in result  # Obtaining itself is NOT in overview
        assert 3 not in result

    def test_character_overview_sections(self):
        blocks = self._make_blocks([
            "Biographical Information",
            "Physical Description",
            "Residence",
            "Gallery",
        ])
        result = _overview_block_indices(blocks, [PageType.CHARACTER])
        assert result == {0, 1, 2}

    def test_monster_overview_sections(self):
        blocks = self._make_blocks(["Battle Statistics", "Drops", "Gallery"])
        result = _overview_block_indices(blocks, [PageType.MONSTER])
        assert result == {0, 1}

    def test_mission_first_rewards_only(self):
        blocks = self._make_blocks([
            "Mission Details", "Rewards", "Walkthrough", "Rewards", "Trivia",
        ])
        result = _overview_block_indices(blocks, [PageType.MISSION])
        assert 0 in result  # Mission Details
        assert 1 in result  # First Rewards
        assert 3 not in result  # Second Rewards NOT in overview

    def test_event_first_rewards_only(self):
        blocks = self._make_blocks([
            "Event Information", "Rewards", "Walkthrough", "Rewards",
        ])
        result = _overview_block_indices(blocks, [PageType.EVENT])
        assert 0 in result
        assert 1 in result
        assert 3 not in result

    def test_duplicate_overview_section_first_only_any_type(self):
        """Any duplicated type-specific overview section: only first in overview."""
        blocks = self._make_blocks([
            "Biographical Information",
            "Residence",
            "Schedule",
            "Residence",
            "Gallery",
        ])
        result = _overview_block_indices(blocks, [PageType.CHARACTER])
        assert 0 in result  # Biographical Information
        assert 1 in result  # First Residence
        assert 3 not in result  # Second Residence NOT in overview

    def test_store_overview_sections(self):
        blocks = self._make_blocks(["Establishment Information", "Stock", "Hours"])
        result = _overview_block_indices(blocks, [PageType.STORE])
        assert 0 in result

    def test_location_overview_sections(self):
        blocks = self._make_blocks([
            "Establishment Information", "Locations", "NPCs",
        ])
        result = _overview_block_indices(blocks, [PageType.LOCATION])
        assert result == {0, 1}

    def test_festival_overview_sections(self):
        blocks = self._make_blocks(["Time", "Information", "Unlock", "Trivia"])
        result = _overview_block_indices(blocks, [PageType.FESTIVAL])
        # Time, Information, Unlock are all in overview
        # Information is also in _GENERAL_OVERVIEW_SECTIONS
        assert result == {0, 1, 2}

    def test_multiple_types_combined(self):
        blocks = self._make_blocks(["Stats", "Obtaining", "Stock", "Gallery"])
        result = _overview_block_indices(blocks, [PageType.ITEM, PageType.STORE])
        # Item: Stats (before Obtaining) → 0
        # Store: nothing extra here (Stock is not "Establishment Information")
        assert 0 in result
        assert 2 not in result

    def test_empty_blocks(self):
        result = _overview_block_indices([], [PageType.GENERIC])
        assert result == set()


# ---------------------------------------------------------------------------
# break_file
# ---------------------------------------------------------------------------


class TestBreakFile:
    def test_item_page_chunks(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = (
            "# Yakmel\nA fluffy animal.\n"
            "## Stats\nHP: 100.\n"
            "## Obtaining\nYou can tame one.\n"
            "## Gallery\n### Screenshots\nSome screenshots."
        )
        _populate_input(input_dir, {"yakmel.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "yakmel.md")

        assert "yakmel-overview.md" in written
        assert "yakmel-obtaining.md" in written
        assert "yakmel-gallery.md" in written
        assert len(written) == 3

        # Overview contains preamble + Stats (before Obtaining)
        overview = (output_dir / "yakmel-overview.md").read_text()
        assert "fluffy animal" in overview
        assert "HP: 100" in overview

    def test_character_page_chunks(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = (
            "# Mi-an\nA young builder.\n"
            "## Biographical Information\nAge: 18.\n"
            "## Physical Description\nShort hair.\n"
            "## Schedule\nGoes to work at 8am.\n"
            "## Gifts\nLikes copper."
        )
        _populate_input(input_dir, {"mi_an.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "mi_an.md")

        assert "mi_an-overview.md" in written
        assert "mi_an-schedule.md" in written
        assert "mi_an-gifts.md" in written

        overview = (output_dir / "mi_an-overview.md").read_text()
        assert "young builder" in overview
        assert "Age: 18" in overview
        assert "Short hair" in overview

    def test_mission_page_first_rewards_in_overview(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = (
            "# Mission: The Bright Sun\nMain quest.\n"
            "## Mission Details\nGo to the ruins.\n"
            "## Rewards\n500 gols.\n"
            "## Walkthrough\nStep 1.\n"
            "## Rewards\nBonus items.\n"
        )
        _populate_input(input_dir, {"mission_the_bright_sun.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "mission_the_bright_sun.md")

        assert "mission_the_bright_sun-overview.md" in written
        assert "mission_the_bright_sun-walkthrough.md" in written
        # Second Rewards is standalone
        assert "mission_the_bright_sun-rewards.md" in written

        overview = (output_dir / "mission_the_bright_sun-overview.md").read_text()
        assert "500 gols" in overview
        assert "Bonus items" not in overview

    def test_l3_subsections_stay_with_l2_parent(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = (
            "## Obtaining\nMain text.\n"
            "### Quest Rewards\nReward text.\n"
            "### Crafting\nCraft text.\n"
            "## Gallery\nImages."
        )
        _populate_input(input_dir, {"table.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "table.md")

        obtaining = (output_dir / "table-obtaining.md").read_text()
        assert "Quest Rewards" in obtaining
        assert "Reward text." in obtaining
        assert "Crafting" in obtaining
        assert "Craft text." in obtaining

    def test_empty_section_skipped(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = "## Gallery\n### Screenshots\nSome text.\n## Trivia\nFun facts."
        _populate_input(input_dir, {"page.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "page.md")

        # Gallery block includes ### Screenshots with text → not empty
        assert "page-gallery.md" in written
        assert "page-trivia.md" in written

    def test_truly_empty_section_skipped(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = "## Empty\n## Trivia\nFun facts."
        _populate_input(input_dir, {"page.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "page.md")

        assert "page-empty.md" not in written
        assert "page-trivia.md" in written

    def test_overview_includes_preamble(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = "# Title\nPreamble text.\n## Section\nSection text."
        _populate_input(input_dir, {"page.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "page.md")

        assert "page-overview.md" in written
        overview = (output_dir / "page-overview.md").read_text()
        assert "Preamble text." in overview

    def test_overview_section_merges_prior_sections(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = (
            "## Stats\nHP: 50.\n"
            "## Overview\nGeneral info.\n"
            "## Gallery\nImages."
        )
        _populate_input(input_dir, {"page.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "page.md")

        overview = (output_dir / "page-overview.md").read_text()
        assert "HP: 50" in overview
        assert "General info" in overview
        assert "Images" not in overview

    def test_creates_output_dir_if_missing(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "deep" / "nested" / "output"
        _populate_input(input_dir, {"a.md": "Some text"})

        breaker = SectionBreaker(input_dir, output_dir)
        breaker.break_file(input_dir / "a.md")

        assert output_dir.is_dir()

    def test_empty_file_produces_no_output(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"empty.md": ""})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "empty.md")

        assert written == []

    def test_no_sections_only_preamble(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"simple.md": "Just plain text."})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "simple.md")

        assert "simple-overview.md" in written
        assert len(written) == 1

    def test_duplicate_l2_titles_numbered(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = "## Rewards\nFirst rewards.\n## Rewards\nSecond rewards."
        _populate_input(input_dir, {"page.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "page.md")

        assert "page-rewards_1.md" in written
        assert "page-rewards_2.md" in written

    def test_store_page_stock_not_in_overview(self, tmp_path):
        """Stock triggers Store type classification but is not an overview section."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = (
            "# By The Stairs\nA shop.\n"
            "## Establishment Information\nLocated downtown.\n"
            "## Stock\n| Item | Price |\n|---|---|\n| Wood | 10 |"
        )
        _populate_input(input_dir, {"by_the_stairs.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "by_the_stairs.md")

        overview = (output_dir / "by_the_stairs-overview.md").read_text()
        assert "Located downtown" in overview
        assert "by_the_stairs-stock.md" in written

    def test_festival_page_chunks(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        content = (
            "# Day of Memories\nA festival.\n"
            "## Time\nSpring 15.\n"
            "## Information\nCelebrate the past.\n"
            "## Unlock\nReach Year 2.\n"
            "## Activities\nDance contest."
        )
        _populate_input(input_dir, {"day_of_memories.md": content})

        breaker = SectionBreaker(input_dir, output_dir)
        written = breaker.break_file(input_dir / "day_of_memories.md")

        overview = (output_dir / "day_of_memories-overview.md").read_text()
        assert "Spring 15" in overview
        assert "Celebrate the past" in overview
        assert "Reach Year 2" in overview
        assert "Dance contest" not in overview
        assert "day_of_memories-activities.md" in written


# ---------------------------------------------------------------------------
# break_all
# ---------------------------------------------------------------------------


class TestBreakAll:
    def test_processes_all_md_files(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "alpha.md": "# Alpha\nContent",
            "beta.md": "# Beta\nContent",
        })

        breaker = SectionBreaker(input_dir, output_dir)
        results = breaker.break_all(max_files=0)

        assert set(results.keys()) == {"alpha.md", "beta.md"}

    def test_skips_existing_output(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "done.md": "# Done\nAlready done",
            "new.md": "# New\nNeeds breaking",
        })
        output_dir.mkdir(parents=True)
        (output_dir / "done-overview.md").write_text("existing", encoding="utf-8")

        breaker = SectionBreaker(input_dir, output_dir)
        results = breaker.break_all(max_files=0)

        assert list(results.keys()) == ["new.md"]

    def test_skips_existing_with_prefix_match(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"yakmel.md": "## Obtaining\nGet one."})
        output_dir.mkdir(parents=True)
        (output_dir / "yakmel-obtaining.md").write_text("x", encoding="utf-8")

        breaker = SectionBreaker(input_dir, output_dir)
        results = breaker.break_all(max_files=0)

        assert results == {}

    def test_force_reprocesses(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {"done.md": "# Done\ncontent"})
        output_dir.mkdir(parents=True)
        (output_dir / "done-overview.md").write_text("old", encoding="utf-8")

        breaker = SectionBreaker(input_dir, output_dir)
        results = breaker.break_all(force=True)

        assert "done.md" in results

    def test_empty_directory_returns_empty(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        breaker = SectionBreaker(input_dir, output_dir)
        results = breaker.break_all()

        assert results == {}

    def test_max_files_limits_processed(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "a.md": "# A\ncontent",
            "b.md": "# B\ncontent",
            "c.md": "# C\ncontent",
            "d.md": "# D\ncontent",
            "e.md": "# E\ncontent",
        })

        breaker = SectionBreaker(input_dir, output_dir)
        results = breaker.break_all(max_files=3)

        assert len(results) == 3

    def test_max_files_zero_processes_all(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "a.md": "# A\ncontent",
            "b.md": "# B\ncontent",
            "c.md": "# C\ncontent",
        })

        breaker = SectionBreaker(input_dir, output_dir)
        results = breaker.break_all(max_files=0)

        assert len(results) == 3

    def test_skipped_files_dont_count_toward_max(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "done.md": "# Done\nalready done",
            "a.md": "# A\ncontent",
            "b.md": "# B\ncontent",
            "c.md": "# C\ncontent",
        })
        output_dir.mkdir(parents=True)
        (output_dir / "done-overview.md").write_text("existing", encoding="utf-8")

        breaker = SectionBreaker(input_dir, output_dir)
        results = breaker.break_all(max_files=2)

        assert len(results) == 2
        assert "done.md" not in results

    def test_continues_on_single_file_failure(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _populate_input(input_dir, {
            "aaa.md": "# A\ngood",
            "ccc.md": "# C\ngood",
        })
        # Create a file that will cause a read error
        bad_path = input_dir / "bbb.md"
        bad_path.mkdir()  # directory instead of file -> read will fail

        breaker = SectionBreaker(input_dir, output_dir)
        results = breaker.break_all(max_files=0)

        assert set(results.keys()) == {"aaa.md", "ccc.md"}
